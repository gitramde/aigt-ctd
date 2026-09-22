"""Read-only frozen baseline access and deterministic same-day target cohorts."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from . import ROOT
from src.phase4.data import BASE,CACHE,guard,prep_config,training_arrays
from src.baseline.preprocess import iter_transformed
from src.baseline.common import json_write,csv_write,require,sha256
OUT=ROOT/'results/temporal_v1'
CONFIG=OUT/'configs';METRICS=OUT/'metrics';MODELS=OUT/'models';FIGURES=OUT/'figures'
PROTOCOL=dict(seed=42,phase='development',lengths=[16,32,64],warmup=63,train_stride=128,evaluation_stride=8,
    batch_size=256,threads=2,max_epochs=3,patience=2,min_improvement=1e-5,learning_rate=0.0003,
    weight_decay=0.01,gradient_clip_norm=1.0,selection_metric='validation_macro_f1_at_0.5',
    small_architecture=dict(d_model=64,layers=2,heads=4,dropout=0.1),
    alternate_architecture=dict(d_model=128,layers=3,heads=4,dropout=0.2),
    architecture_search='three lengths at small architecture; one alternate architecture at validation-best length',
    ablation='same architecture and target records; only final flow with same final positional index',
    attention='bidirectional within historical window including current flow; no later flow is present',
    optimizer='AdamW',feedforward_multiplier=2,position_encoding='fixed sinusoidal',feature_dtype='float32')

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(name,rows): csv_write(OUT/name,rows)
def cohort_indices(day_spans,stride,warmup=63):
    return np.concatenate([np.arange(s+warmup,e,stride,dtype=np.int64) for s,e in day_spans])
def integrity(full=False):
    guard(full=full)
    manifest=read(METRICS/'baseline_snapshot.json')
    for row in manifest: require(sha256(row['path'])==row['sha256'],'Frozen baseline/Phase 4/4A artifact changed: '+row['path'])
    require(read(CONFIG/'protocol.json')==PROTOCOL,'Temporal protocol changed')

def prepare():
    for d in (CONFIG,METRICS,MODELS,FIGURES): d.mkdir(parents=True,exist_ok=True)
    if not (CONFIG/'protocol.json').exists(): json_write(CONFIG/'protocol.json',PROTOCOL)
    require(read(CONFIG/'protocol.json')==PROTOCOL,'Refusing to overwrite an existing protocol')
    if not (METRICS/'baseline_snapshot.json').exists():
        guard(full=True)
        files=[p for p in BASE.rglob('*') if p.is_file()]
        for directory in ('src/baseline','src/phase4','src/phase4a'):
            files+=list((ROOT/directory).glob('*.py'))
        files+=[ROOT/'configs/baseline.json',ROOT/'configs/phase4_seed42.json',CACHE/'training_cache.json']
        json_write(METRICS/'baseline_snapshot.json',[dict(path=str(p),sha256=sha256(p)) for p in files])
    integrity(full=True)
    cache=read(CACHE/'training_cache.json')
    for key in ('x','y'): require(sha256(cache[key+'_path'])==cache[key+'_sha256'],'Training cache changed')
    if (METRICS/'sequence_manifest.json').exists():
        for part,item in read(METRICS/'sequence_manifest.json')['partitions'].items():
            require(sha256(item['cohort_path'])==item['cohort_sha256'],'Sequence cohort changed')
        return
    plan=read(BASE/'split_plan.json');state=read(BASE/'preprocessing.json')
    book=read(BASE/'metrics/pretraining_check.json')['diagnostic_codebook'];mapping={s:i for i,s in enumerate(book)}
    rows=[];manifest=dict(codebook=book,features=state['output_feature_names'],partitions={},checks={})
    previous_partition_max=None
    for part in ('train','validation','test'):
        entry=plan['partitions'][part];n=entry['rows'];spans=[];offset=0
        for file in entry['files']:
            spans.append((offset,offset+file['rows']));offset+=file['rows']
        require(offset==n,'Day/partition count mismatch')
        stride=PROTOCOL['train_stride'] if part=='train' else PROTOCOL['evaluation_stride']
        ids=cohort_indices(spans,stride);size=len(ids)
        y=np.empty(size,dtype=np.uint8);family=np.empty(size,dtype=np.uint8);source=np.empty(size,dtype=np.int64)
        times=np.empty(size,dtype='datetime64[us]');days=np.empty(size,dtype='datetime64[D]');files=np.empty(size,dtype=object)
        offset=0;last=None
        for b in pq.ParquetFile(entry['membership_path']).iter_batches(batch_size=50000):
            ts=b.column(b.schema.get_field_index('timestamp')).to_numpy(zero_copy_only=False).astype('datetime64[us]')
            require(not np.isnat(ts).any() and (ts.astype('datetime64[Y]')!=np.datetime64('1970','Y')).all(),'Invalid temporal timestamp')
            require((ts[1:]>=ts[:-1]).all() and (last is None or ts[0]>=last),'Nonchronological frozen membership')
            last=ts[-1]
            for i,(start,end) in enumerate(spans):
                lo=max(start,offset);hi=min(end,offset+len(b))
                if lo<hi: require((ts[lo-offset:hi-offset].astype('datetime64[D]')==np.datetime64(entry['dates'][i])).all(),'Day boundary mismatch')
            lo=np.searchsorted(ids,offset);hi=np.searchsorted(ids,offset+len(b));local=ids[lo:hi]-offset
            sub=b.take(pa.array(local))
            y[lo:hi]=sub.column(sub.schema.get_field_index('target_binary')).to_numpy()
            attack=sub.column(sub.schema.get_field_index('attack_label')).to_pylist();family[lo:hi]=[mapping[a] for a in attack]
            require(np.array_equal(y[lo:hi],[int(a!='Benign') for a in attack]),'Target/family mismatch')
            source[lo:hi]=sub.column(sub.schema.get_field_index('source_row')).to_numpy()
            files[lo:hi]=sub.column(sub.schema.get_field_index('source_file')).to_pylist()
            times[lo:hi]=ts[local];days[lo:hi]=ts[local].astype('datetime64[D]');offset+=len(b)
        require(offset==n,'Incomplete membership scan')
        if previous_partition_max is not None: require(np.datetime64(entry['timestamp_min'])>previous_partition_max,'Partitions overlap in time')
        previous_partition_max=np.datetime64(entry['timestamp_max'])
        for length in PROTOCOL['lengths']:
            for i,(start,end) in enumerate(spans):
                targets=ids[(ids>=start)&(ids<end)]
                require(((targets-length+1)>=start).all() and (targets<end).all(),'Sequence boundary violation')
                for code,label in enumerate(book):
                    mask=(ids>=start)&(ids<end)&(family==code)
                    rows.append(dict(partition=part,date=entry['dates'][i],length=length,stride=stride,attack_family=label,
                        sequence_targets=int(mask.sum()),day_rows=end-start,excluded_prefix_rows=min(63,end-start),
                        source_file=entry['files'][i]['filename'],target_rule='final flow only'))
        path=METRICS/f'{part}_cohort.parquet'
        pq.write_table(pa.table(dict(partition_index=ids,target_binary=y,family_code=family,source_row=source,
            source_file=files.tolist(),timestamp=pa.array(times),file_date=pa.array(days))),path,compression='zstd')
        manifest['partitions'][part]=dict(rows=n,targets=size,stride=stride,spans=spans,dates=entry['dates'],
            benign=int((y==0).sum()),malicious=int(y.sum()),coverage_fraction=size/n,
            class_counts={label:int((family==i).sum()) for i,label in enumerate(book)},cohort_path=str(path),cohort_sha256=sha256(path))
        print('SEQUENCE COHORT',part,size,manifest['partitions'][part]['class_counts'],flush=True)
    manifest['checks']=dict(no_partition_crossing=True,no_day_crossing=True,chronological_order_verified=True,
        no_1970=True,no_future_flow=True,common_target_records_across_lengths=True,labels_metadata_only=True)
    json_write(METRICS/'sequence_manifest.json',manifest);write('sequence_counts.csv',rows)
    json_write(METRICS/'pretraining_integrity.json',dict(status='PASS',raw_and_cleaned_hashes_verified=True,training_cache_verified=True,baseline_snapshot_sha256=sha256(METRICS/'baseline_snapshot.json')))

def cohort(part):
    t=pq.read_table(METRICS/f'{part}_cohort.parquet')
    return {k:t[k].to_numpy(zero_copy_only=False) for k in ('partition_index','target_binary','family_code')}

def features(part):
    if part=='train': return training_arrays()[0]
    n=read(METRICS/'sequence_manifest.json')['partitions'][part]['rows'];d=len(read(BASE/'preprocessing.json')['output_feature_names'])
    x=np.empty((n,d),dtype=np.float32);offset=0;digest=hashlib.sha256()
    for xb,y,metadata in iter_transformed(prep_config(),part,scaled=True):
        digest.update(np.ascontiguousarray(xb,dtype='<f8').tobytes());x[offset:offset+len(xb)]=xb;offset+=len(xb)
    with (BASE/'preprocessing_application_audit.csv').open(newline='') as f:
        expected=next(r['transformed_float64_sha256'] for r in csv.DictReader(f) if r['partition']==part)
    require(offset==n and digest.hexdigest()==expected,'Frozen transform mismatch')
    return x

def sequence_batches(x,ids,length,batch_size=256,order=None):
    sequence_order=np.arange(len(ids)) if order is None else order
    for start in range(0,len(ids),batch_size):
        positions=sequence_order[start:start+batch_size];endpoints=ids[positions]
        indices=endpoints[:,None]+np.arange(1-length,1)
        yield positions,np.ascontiguousarray(x[indices],dtype=np.float32)

if __name__=='__main__': prepare()
