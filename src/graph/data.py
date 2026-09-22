"""Approved, train-fitted Feb-20 arrays and causal window prefixes."""
import time
import hashlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .common import *
from src.baseline.preprocess import fit_training,FrozenPreprocessor


def timestamp_units(values):
    """Arrow timestamps may be microseconds, not pandas' historical ns default."""
    microseconds=np.asarray(values,dtype='datetime64[us]').astype(np.int64)
    return microseconds//1_000_000,microseconds


def prepare():
    initialize()
    if (METRICS/'data_manifest.json').exists():
        for row in read(METRICS/'data_manifest.json')['artifacts']:
            require(sha256(row['path'])==row['sha256'],'Graph cache changed')
        print('Graph data already verified',flush=True);return
    start=time.perf_counter()
    proposal=read(OUT/'pre_split/proposed_split.json')
    source=proposal['inputs']
    for kind in ('raw','cleaned'):
        require(sha256(source[kind+'_path'])==source[kind+'_sha256'],'Frozen Feb-20 input mismatch')
    metadata_path=CACHE/'metadata.parquet'
    if not metadata_path.exists():
        meta=pq.read_table(source['cleaned_path'],columns=['source_row','timestamp','target_binary','Flow Duration','TotLen Fwd Pkts','TotLen Bwd Pkts','Dst Port','Protocol']).to_pandas()
        meta=meta.rename(columns={'Flow Duration':'duration','Dst Port':'dst_port','Protocol':'protocol'})
        meta['bytes']=meta.pop('TotLen Fwd Pkts').fillna(0)+meta.pop('TotLen Bwd Pkts').fillna(0)
        n=len(meta); raw_count=int(meta.source_row.max())
        lookup=np.full(raw_count+1,-1,dtype=np.int32);lookup[meta.source_row]=np.arange(n,dtype=np.int32)
        src=np.empty(n,dtype=np.int32);dst=np.empty(n,dtype=np.int32);sport=np.empty(n,dtype=np.int32)
        vocabulary={};offset=0;matched=0
        columns=['Src IP','Dst IP','Src Port']
        reader=pcsv.open_csv(source['raw_path'],read_options=pcsv.ReadOptions(block_size=32*1024*1024),
            convert_options=pcsv.ConvertOptions(include_columns=columns,column_types={c:pa.string() for c in columns}))
        for batch in reader:
            ids=np.arange(offset+1,offset+1+len(batch));offset+=len(batch)
            valid=ids<=raw_count; ids=ids[valid];positions=lookup[ids];keep=positions>=0
            d=batch.to_pandas().loc[valid].loc[keep];positions=positions[keep]
            for col,dest in [('Src IP',src),('Dst IP',dst)]:
                values=d[col];new=values.unique()
                for ip in new:
                    require(isinstance(ip,str) and bool(ip),'Missing IP')
                    if ip not in vocabulary: vocabulary[ip]=len(vocabulary)
                dest[positions]=values.map(vocabulary).to_numpy(dtype=np.int32)
            sport[positions]=d['Src Port'].astype(np.int32);matched+=len(d)
            if offset//2_000_000 != (offset-len(batch))//2_000_000: print('Endpoint join',offset,flush=True)
        require(matched==n,'Endpoint join incomplete')
        meta['src']=src;meta['dst']=dst;meta['src_port']=sport
        meta['protocol']=pd.to_numeric(meta.protocol).astype(np.int16)
        meta['partition']=np.where(meta.timestamp<pd.Timestamp(proposal['boundaries'][0]),0,
            np.where(meta.timestamp<pd.Timestamp(proposal['boundaries'][1]),1,2)).astype(np.int8)
        seconds,microseconds=timestamp_units(meta.timestamp)
        meta['second']=seconds
        require(meta.duration.notna().all() and (meta.duration>=0).all(),'Invalid flow durations; cannot establish completion time')
        meta['completion_us']=microseconds+meta.duration.astype(np.int64)
        meta=meta.drop(columns=['timestamp','duration'])
        for part,r in zip(PARTS,proposal['partitions']):
            values=meta.loc[meta.partition==PARTS.index(part),'source_row'].to_numpy(dtype='<i8')
            require(hashlib.sha256(values.tobytes()).hexdigest()==r['source_row_sequence_sha256'],'Approved membership mismatch')
        pq.write_table(pa.Table.from_pandas(meta,preserve_index=False),metadata_path,compression='zstd')
        del meta,lookup,src,dst,sport,vocabulary
    meta=pq.read_table(metadata_path).to_pandas();n=len(meta)
    require((pd.to_datetime(meta.second,unit='s').dt.date==pd.Timestamp('2018-02-20').date()).all(),'Cached timestamps lost their calendar date')
    ids=[];cohort=[]
    for minute,group in meta.groupby(meta.second//60,sort=True):
        require(group.partition.nunique()==1,'Target minute crosses partition')
        candidates=group.index[group.second%60>=PROTOCOL['cutoff_second']].to_numpy()
        if len(candidates)==0: continue
        part=int(group.partition.iloc[0]);stride=PROTOCOL['train_stride'] if part==0 else PROTOCOL['evaluation_stride']
        targets=candidates[::stride];ids.extend(targets)
        cohort.append(dict(minute=int(minute),partition=PARTS[part],targets=len(targets),start=len(ids)-len(targets),end=len(ids)))
    ids=np.asarray(ids,dtype=np.int64);np.save(CACHE/'target_ids.npy',ids)
    require({g['partition'] for g in cohort}==set(PARTS),'Missing target partition')
    json_write(CACHE/'target_groups.json',cohort)
    pq.write_table(pa.Table.from_pandas(meta.iloc[ids][['source_row','second','target_binary','partition']],preserve_index=False),METRICS/'evaluation_cohort.parquet',compression='zstd')
    del meta
    def training_batches():
        for batch in pq.ParquetFile(source['cleaned_path']).iter_batches(batch_size=50000):
            t=batch.column(batch.schema.get_field_index('timestamp'))
            b=batch.filter(pc.less(t,pa.scalar(pd.Timestamp(proposal['boundaries'][0]).to_pydatetime(),type=t.type)))
            if len(b): yield b
    path=CONFIG/'preprocessing.json'
    if not path.exists():
        print('Fitting Feb-20 TRAIN-only preprocessing',flush=True)
        frozen=read(ROOT/'results/baseline_v1/preprocessing.json')
        features=frozen['numeric_columns']+frozen['categorical_columns']
        state=fit_training(training_batches,features,['Protocol'])
        require(state['training_rows']==proposal['partitions'][0]['rows'],'Fit included wrong rows')
        state.update(split_approval_sha256=sha256(CONFIG/'split_approval.json'),fit_scope='approved Feb-20 TRAIN only')
        json_write(path,state)
    processor=FrozenPreprocessor.load(path);names=list(processor.output_names)
    hnames=[name for name in PROTOCOL['historical_edge_feature_names'] if name in names]
    require(len(hnames)>=6,'Historical edge features unavailable')
    hcols=[names.index(name) for name in hnames]
    x=np.lib.format.open_memmap(CACHE/'target_features.npy',mode='w+',dtype=np.float32,shape=(len(ids),len(names)))
    hx=np.lib.format.open_memmap(CACHE/'history_features.npy',mode='w+',dtype=np.float32,shape=(n,len(hcols)))
    offset=0
    for batch in pq.ParquetFile(source['cleaned_path']).iter_batches(batch_size=50000):
        features,_=processor.transform(batch);stop=offset+len(batch)
        lo,hi=np.searchsorted(ids,[offset,stop]);x[lo:hi]=features[ids[lo:hi]-offset]
        hx[offset:stop]=features[:,hcols];offset=stop
    require(offset==n,'Incomplete transform');x.flush();hx.flush();del x,hx
    files=[metadata_path,CACHE/'target_ids.npy',CACHE/'target_groups.json',CACHE/'target_features.npy',CACHE/'history_features.npy',METRICS/'evaluation_cohort.parquet',path]
    json_write(METRICS/'data_manifest.json',dict(status='PASS',rows=n,target_rows=len(ids),edge_feature_names=names,history_feature_names=hnames,
        preprocessing_sha256=sha256(path),split_approval_sha256=sha256(CONFIG/'split_approval.json'),
        preparation_seconds=time.perf_counter()-start,artifacts=[dict(path=str(p),sha256=sha256(p)) for p in files]))
    print('Train-fitted graph arrays complete',len(ids),'targets',flush=True)


class GraphData:
    def __init__(self):
        self.meta=pq.read_table(CACHE/'metadata.parquet').to_pandas()
        self.ids=np.load(CACHE/'target_ids.npy');self.groups=read(CACHE/'target_groups.json')
        self.x=np.load(CACHE/'target_features.npy',mmap_mode='r');self.hx=np.load(CACHE/'history_features.npy',mmap_mode='r')
        self.times=self.meta.second.to_numpy();self.y=self.meta.target_binary.to_numpy()[self.ids]
        self.part=self.meta.partition.to_numpy()[self.ids]
        self.src=self.meta.src.to_numpy();self.dst=self.meta.dst.to_numpy()

    def snapshot(self,g,window):
        cutoff=g['minute']*60+PROTOCOL['cutoff_second'];start=(cutoff//(window*60))*(window*60)
        lo,hi=np.searchsorted(self.times,[start,cutoff]);history=np.arange(lo,hi,dtype=np.int64)
        history=history[self.meta.completion_us.to_numpy()[history]<cutoff*1_000_000]
        require((self.meta.partition.to_numpy()[history]==PARTS.index(g['partition'])).all(),'History crosses partition')
        target_positions=np.arange(g['start'],g['end']);targets=self.ids[target_positions]
        require((self.times[targets]>=cutoff).all(),'Target before cutoff')
        require((self.times[targets]<(start+window*60)).all(),'Target outside calendar window')
        nodes=np.unique(np.r_[self.src[history],self.dst[history],self.src[targets],self.dst[targets]])
        hs=np.searchsorted(nodes,self.src[history]);hd=np.searchsorted(nodes,self.dst[history])
        nf=np.zeros((len(nodes),8),dtype=np.float32)
        b=np.maximum(self.meta.bytes.to_numpy()[history],0)
        for col,idx,weights in [(0,hd,None),(1,hs,None),(2,hd,b),(3,hs,b)]:
            nf[:,col]=np.bincount(idx,weights=weights,minlength=len(nodes))
        for protocol,col in [(6,4),(17,5)]:
            mask=self.meta.protocol.to_numpy()[history]==protocol
            nf[:,col]=np.bincount(np.r_[hs[mask],hd[mask]],minlength=len(nodes))
        nf[:,6]=np.bincount(hs[self.meta.src_port.to_numpy()[history]<1024],minlength=len(nodes))
        nf[:,7]=np.bincount(hd[self.meta.dst_port.to_numpy()[history]<1024],minlength=len(nodes))
        nf=np.log1p(nf)
        if len(history)>PROTOCOL['history_edge_cap']:
            order=np.lexsort((self.meta.source_row.to_numpy()[history],self.meta.completion_us.to_numpy()[history]))
            chosen=order[-PROTOCOL['history_edge_cap']:]
        else: chosen=np.arange(len(history))
        ts=np.searchsorted(nodes,self.src[targets]);td=np.searchsorted(nodes,self.dst[targets])
        # Nodes disconnected from every retained message edge and target cannot
        # affect a prediction. Pruning them preserves the full-history aggregates.
        used=np.unique(np.r_[hs[chosen],hd[chosen],ts,td])
        return dict(nodes=nf[used],edges=np.vstack([np.searchsorted(used,hs[chosen]),np.searchsorted(used,hd[chosen])]),edge_features=np.asarray(self.hx[history[chosen]]),
            target_src=np.searchsorted(used,ts),target_dst=np.searchsorted(used,td),
            x=np.asarray(self.x[target_positions]),y=self.y[target_positions],positions=target_positions,
            history_rows=history[chosen],history_count=len(history),cutoff=cutoff,window_start=start)


def diagnostics():
    data=GraphData();meta=data.meta;rows=[];prefix=[]
    for window in (1,5,10):
        for bucket,group in meta.groupby(meta.second//(window*60),sort=True):
            src=group.src.to_numpy();dst=group.dst.to_numpy();nodes=np.unique(np.r_[src,dst]);n=len(nodes);e=len(group)
            pairs,counts=np.unique(np.column_stack([src,dst]),axis=0,return_counts=True)
            s=np.searchsorted(nodes,pairs[:,0]);d=np.searchsorted(nodes,pairs[:,1])
            adjacency=coo_matrix((np.ones(len(pairs),dtype=np.uint8),(s,d)),shape=(n,n)).tocsr()
            components=connected_components(adjacency,directed=True,connection='weak',return_labels=False)
            strong=connected_components(adjacency,directed=True,connection='strong',return_labels=False)
            nonself=int((pairs[:,0]!=pairs[:,1]).sum())
            rows.append(dict(window_minutes=window,window_start=str(pd.Timestamp(int(bucket)*window*60,unit='s')),
                partition=PARTS[int(group.partition.iloc[0])],nodes=n,edges=e,weak_connected_components=int(components),strong_connected_components=int(strong),
                isolated_nodes=0,unique_directed_pairs=len(pairs),repeated_directed_pairs=int((counts>1).sum()),extra_parallel_edges=int((counts-1).sum()),
                density=nonself/(n*(n-1)) if n>1 else 0.,benign=int((group.target_binary==0).sum()),malicious=int(group.target_binary.sum())))
        for g in data.groups:
            snap=data.snapshot(g,window)
            prefix.append(dict(window_minutes=window,minute=g['minute'],partition=g['partition'],target_edges=len(snap['y']),
                history_edges=snap['history_count'],message_edges=len(snap['history_rows']),nodes=len(snap['nodes']),
                cold_target_endpoints=int(((snap['nodes'][np.r_[snap['target_src'],snap['target_dst']],:2]).sum(axis=1)==0).sum())))
        print('Graph diagnostics complete',window,'minutes',flush=True)
    write('graph_snapshot_diagnostics.csv',rows);write('causal_prefix_diagnostics.csv',prefix)
    summary=[]
    for window in (1,5,10):
        for part in PARTS:
            subset=[r for r in rows if r['window_minutes']==window and r['partition']==part]
            for metric in ('nodes','edges','weak_connected_components','strong_connected_components','isolated_nodes','repeated_directed_pairs','density','benign','malicious'):
                values=np.asarray([r[metric] for r in subset]);summary.append(dict(window_minutes=window,partition=part,snapshots=len(subset),metric=metric,
                    mean=float(values.mean()),median=float(np.median(values)),p90=float(np.percentile(values,90)),maximum=float(values.max())))
    write('graph_diagnostics_summary.csv',summary)

if __name__=='__main__':
    prepare();diagnostics()
