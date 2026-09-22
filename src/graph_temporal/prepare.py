"""Pretraining integrity and completion-causal temporal coverage gate."""
import time
from collections import defaultdict
from . import ROOT
import numpy as np
import pyarrow.parquet as pq
from src.baseline.common import sha256,json_write,csv_write,require
from src.graph.common import read,guard

OUT=ROOT/'results/graph_temporal_v1'
GRAPH=ROOT/'results/graph_v1'
CACHE=ROOT/'data/graph_v1'
NAME='gatv2_w1_h128_p2'
PARTS=('train','validation','test')
PROTOCOL=dict(seed=42,encoder=NAME,encoder_frozen=True,window_minutes=1,hidden=128,gat_layers=2,gat_heads=4,gat_dropout=.2,
    lengths=[1,4,8],d_model=64,layers=2,heads=4,dropout=.1,epochs=4,patience=2,learning_rate=.0005,weight_decay=.01,
    batch_size=2048,threads=2,selection='validation macro-F1 at .5; min improvement .00001',
    temporal_step='one previous calendar minute; L includes current step and L-1 preceding minutes; no compression of missing minutes',
    historical_observation='one completed Phase6 sampled target flow representation from an earlier minute; frozen own-minute GAT context and own complete flow vector',
    eligibility='same partition; prior minute; completion strictly before current minute second30 cutoff; source_row breaks completion ties',
    identity='directed pair first, same source endpoint second, same destination endpoint third; latest eligible completion within that minute',
    padding='missing historical slots zero padded and key-padding masked; current always present; learned absolute slot positions shared across L1/4/8',
    coverage_gate='Diagnostic-stage revised TRAIN-only gate: >=10000 fully populated L4 and L8 contexts; missing contexts masked for all remaining targets; no test metrics used',
    representation='concat frozen GAT source node embedding, destination node embedding, current full flow features; historical tokens belong to earlier completed observed flows',
    scope='sampled-target history; no labels, raw endpoint ID features, cross-partition tokens or future snapshots')

def snapshot_prior():
    guard()
    expected=[]
    expected+=read(GRAPH/'metrics/data_manifest.json')['artifacts']
    for name,digest in read(GRAPH/'metrics/integrity_verification.json')['source_code_sha256'].items():
        expected.append(dict(path=str(ROOT/'src/graph'/name),sha256=digest))
    source=read(GRAPH/'pre_split/proposed_split.json')['inputs']
    expected += [dict(path=source[k+'_path'],sha256=source[k+'_sha256']) for k in ('raw','cleaned')]
    lock=read(GRAPH/'configs/threshold_lock.json')
    expected += [dict(path=str(GRAPH/'selected_model.json'),sha256=lock['selected_model_sha256']),
                 dict(path=str(GRAPH/'thresholds.csv'),sha256=lock['thresholds_sha256'])]
    for name in (NAME,'edge_mlp'):
        entry=lock['artifacts'][name]
        expected.append(dict(path=entry['model_path'],sha256=entry['model_sha256']))
        expected.append(dict(path=str(GRAPH/f'metrics/{name}_training.json'),sha256=entry['training_record_sha256']))
        for part in read(GRAPH/f'metrics/{name}_evaluation.json')['partitions'].values():
            expected.append(dict(path=part['prediction_path'],sha256=part['prediction_sha256']))
    for item in expected:
        require(sha256(item['path'])==item['sha256'],'Phase6 hash mismatch: '+item['path'])
    files=[]
    for folder in ('results/baseline_v1','results/temporal_v1','results/graph_v1','results/anomaly_v1','results/fusion_v1',
                   'src/baseline','src/phase4','src/phase4a','src/temporal','src/graph','src/anomaly','src/fusion'):
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts: files.append(dict(path=str(p),sha256=sha256(p)))
    json_write(OUT/'metrics/protected_artifacts.json',files)
    json_write(OUT/'metrics/phase6_hash_verification.json',dict(status='PASS',verified=expected))
    print('Phase6 hashes verified; prior artifacts protected',flush=True)

def build_history(meta,ids,timings=None):
    targets=meta.iloc[ids].reset_index(drop=True)
    n=len(ids);history=np.full((n,8),-1,dtype=np.int64);types=np.zeros((n,8),dtype=np.uint8)
    counts=np.zeros((n,3),dtype=np.int32)
    minute=targets.second.to_numpy()//60;completion=targets.completion_us.to_numpy()
    src=targets.src.to_numpy();dst=targets.dst.to_numpy();part=targets.partition.to_numpy();row=targets.source_row.to_numpy()
    # Per partition/minute keyed completed observations. Lookup never crosses a minute or partition.
    for partition in range(3):
        started=time.perf_counter()
        past={};groups={m:np.flatnonzero((part==partition)&(minute==m)) for m in np.unique(minute[part==partition])}
        for m,positions in sorted(groups.items()):
            cutoff=(int(m)*60+30)*1_000_000
            maps={};totals=[defaultdict(int),defaultdict(int),defaultdict(int)]
            for old,oldpositions in past.items():
                eligible=oldpositions[completion[oldpositions]<cutoff]
                pair={};source={};dest={}
                for pos in eligible[np.lexsort((row[eligible],completion[eligible]))]:
                    pair[(int(src[pos]),int(dst[pos]))]=int(pos);source[int(src[pos])]=int(pos);dest[int(dst[pos])]=int(pos)
                maps[old]=(pair,source,dest)
                for total,lookup in zip(totals,(pair,source,dest)):
                    for key in lookup: total[key]+=1
            for pos in positions:
                key=(int(src[pos]),int(dst[pos]))
                counts[pos]=[totals[0][key],totals[1][key[0]],totals[2][key[1]]]
                for lag in range(1,9):
                    lookups=maps.get(m-lag)
                    if lookups is None:continue
                    pairs,sources,dests=lookups
                    choices=(pairs.get(key,-1),sources.get(key[0],-1),dests.get(key[1],-1))
                    for kind,choice in enumerate(choices,1):
                        if choice>=0:
                            history[pos,lag-1]=choice;types[pos,lag-1]=kind;break
            past[m]=positions
        if timings is not None:
            timings.append(dict(partition=PARTS[partition],history_index_seconds=time.perf_counter()-started,
                                targets=int((part==partition).sum())))
    return targets,history,types,counts

def coverage(targets,history,types,counts):
    rows=[]
    for partition,part in enumerate(PARTS):
        for label in ('all','Benign','DDoS attacks-LOIC-HTTP'):
            mask=targets.partition.to_numpy()==partition
            if label!='all': mask &= targets.target_binary.to_numpy()==int(label!='Benign')
            n=int(mask.sum());r=dict(partition=part,label=label,targets=n)
            for j,name in enumerate(('pair','source','destination')):
                c=counts[mask,j]
                for k in (1,4,8):
                    r[f'{name}_ge{k}_count']=int((c>=k).sum());r[f'{name}_ge{k}_percent']=100*float(np.mean(c>=k)) if n else 0.
                for stat,value in [('median',np.median(c) if n else 0),('p90',np.percentile(c,90) if n else 0),('maximum',c.max() if n else 0)]: r[f'{name}_history_{stat}']=float(value)
            for steps in (3,7,8):
                c=(history[mask,:steps]>=0).sum(1)
                r[f'previous_{steps}_slots_complete_percent']=100*float(np.mean(c==steps)) if n else 0.
                r[f'previous_{steps}_slots_any_percent']=100*float(np.mean(c>0)) if n else 0.
            for kind,name in ((1,'pair'),(2,'source_fallback'),(3,'destination_fallback')):
                r[f'previous7_{name}_tokens']=int((types[mask,:7]==kind).sum())
            rows.append(r)
    csv_write(OUT/'temporal_history_coverage.csv',rows)
    train=next(r for r in rows if r['partition']=='train' and r['label']=='all')
    passed=all(train[f'previous_{steps}_slots_complete_percent']*train['targets']/100>=10000 for steps in (3,7))
    json_write(OUT/'metrics/coverage_gate.json',dict(passed=passed,rule=PROTOCOL['coverage_gate'],training=train))
    return passed,rows

def main():
    json_write(OUT/'configs/protocol.json',PROTOCOL)
    snapshot_prior()
    meta=pq.read_table(CACHE/'metadata.parquet').to_pandas();ids=np.load(CACHE/'target_ids.npy')
    targets,history,types,counts=build_history(meta,ids)
    # Each historical token represents a completed actual earlier flow with matching entity.
    for lag in range(8):
        valid=history[:,lag]>=0;p=np.flatnonzero(valid);q=history[valid,lag]
        require((targets.partition.to_numpy()[p]==targets.partition.to_numpy()[q]).all(),'Cross-partition history')
        require((targets.second.to_numpy()[q]//60==targets.second.to_numpy()[p]//60-lag-1).all(),'Invalid calendar step')
        require((targets.completion_us.to_numpy()[q]<(targets.second.to_numpy()[p]//60*60+30)*1_000_000).all(),'Future completion in history')
    np.save(OUT/'metrics/history_indices.npy',history);np.save(OUT/'metrics/history_identity_types.npy',types)
    np.save(OUT/'metrics/eligible_target_ids.npy',ids)
    np.save(OUT/'metrics/eligible_source_rows.npy',targets.source_row.to_numpy())
    passed,rows=coverage(targets,history,types,counts)
    json_write(OUT/'matched_cohort_manifest.json',dict(status='coverage_pass' if passed else 'stopped_coverage',
        all_phase6_targets_retained=True,rows=len(ids),partitions={p:int((targets.partition==i).sum()) for i,p in enumerate(PARTS)},
        phase6_cohort_sha256=sha256(GRAPH/'metrics/evaluation_cohort.parquet'),
        artifacts=[dict(path=str(p),sha256=sha256(p)) for p in (OUT/'metrics').glob('*.npy')]))
    print('COVERAGE GATE',passed,flush=True)
    for r in rows:
        print(r['partition'],r['label'],'pair >=1/4/8',*[round(r[f'pair_ge{k}_percent'],2) for k in (1,4,8)],
              'full L4/L8',round(r['previous_3_slots_complete_percent'],2),round(r['previous_7_slots_complete_percent'],2),flush=True)

if __name__=='__main__': main()
