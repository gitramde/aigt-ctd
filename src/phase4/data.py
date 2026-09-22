"""Read-only frozen-data checks, diagnostic labels and a model-input cache."""
import argparse
import csv
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from . import ROOT
from src.baseline.common import csv_write,json_write,load_config,require,sha256
from src.baseline.preprocess import iter_transformed,load_plan

BASE=ROOT/'results/baseline_v1'
METRICS=BASE/'metrics'
MODELS=BASE/'models'
FIGURES=BASE/'figures'
CACHE=ROOT/'data/phase4_cache'
MODEL_NAMES=('logistic_regression','random_forest','xgboost','mlp')


def settings():
    return json.loads((ROOT/'configs/phase4_seed42.json').read_text())


def prep_config():
    return load_config(ROOT/'configs/baseline.json')


def guard(full=False):
    saved=json.loads((METRICS/'frozen_input_guard.json').read_text())
    checks=saved['small_artifacts']+ (saved['data_artifacts'] if full else [])
    for row in checks:
        require(sha256(row['path'])==row['sha256'],f'Frozen artifact changed: {row["path"]}')


def preflight():
    for folder in (METRICS,MODELS,FIGURES,CACHE):
        folder.mkdir(parents=True,exist_ok=True)
    if (METRICS/'pretraining_check.json').exists():
        guard(full=True)
        return
    cfg=prep_config()
    _,manifest,plan=load_plan(cfg)
    verification=json.loads((BASE/'verification.json').read_text())
    require(verification['status']=='PASS','Frozen preparation has not passed verification')
    for name,digest in verification['implementation_sha256'].items():
        require(sha256(ROOT/'src/baseline'/name)==digest,f'Frozen preparation implementation changed: {name}')
    small=[dict(path=str(p),sha256=sha256(p)) for p in sorted(BASE.iterdir()) if p.is_file()]
    small += [dict(path=str(p),sha256=sha256(p)) for p in sorted((ROOT/'src/baseline').glob('*.py'))]
    small.append(dict(path=str(ROOT/'configs/baseline.json'),sha256=sha256(ROOT/'configs/baseline.json')))
    artifacts=[]
    for file in manifest['files']:
        print('Preflight hashes:',file['filename'],flush=True)
        for path_key,hash_key in [('raw_path','raw_sha256'),('cleaned_path','cleaned_sha256'),('non_temporal_path','non_temporal_sha256')]:
            if path_key in file:
                require(sha256(file[path_key])==file[hash_key],f'Frozen data mismatch: {file[path_key]}')
                artifacts.append(dict(path=file[path_key],sha256=file[hash_key]))
    class_rows=[]; binary_rows=[]; distributions={}
    for part in ('train','validation','test'):
        entry=plan['partitions'][part]
        path=entry['membership_path']
        require(sha256(path)==entry['membership_sha256'],'Membership changed')
        artifacts.append(dict(path=path,sha256=entry['membership_sha256']))
        labels=Counter(); binary=Counter()
        for batch in pq.ParquetFile(path).iter_batches(columns=['attack_label','target_binary'],batch_size=100000):
            attack=batch.column(0).to_pylist()
            y=batch.column(1).to_numpy()
            require(np.array_equal(y,np.asarray([int(v!='Benign') for v in attack])),'Target/diagnostic label mismatch')
            labels.update(attack)
            binary.update(map(int,y))
        require(dict(labels)==entry['class_counts'],'Class counts differ from frozen split')
        distributions[part]=labels
        binary_rows.append(dict(partition=part,benign=binary[0],malicious=binary[1],records=sum(binary.values())))
    all_labels=sorted(set().union(*(set(c) for c in distributions.values())))
    for part,labels in distributions.items():
        for label in all_labels:
            class_rows.append(dict(partition=part,attack_family=label,records=labels[label],
                                   present=labels[label]>0,training_status='KNOWN' if distributions['train'][label]>0 else 'UNSEEN'))
    csv_write(METRICS/'pretraining_binary_counts.csv',binary_rows)
    csv_write(METRICS/'pretraining_attack_family_counts.csv',class_rows)
    unseen={p:sorted(set(distributions[p])-set(distributions['train'])) for p in ('validation','test')}
    json_write(METRICS/'frozen_input_guard.json',dict(small_artifacts=small,data_artifacts=artifacts))
    json_write(METRICS/'pretraining_check.json',dict(status='PASS',binary_counts=binary_rows,
               attack_family_counts={p:dict(c) for p,c in distributions.items()},unseen_families=unseen,
               diagnostic_codebook=all_labels,target='0 benign; 1 malicious',
               preprocessing_sha256=sha256(BASE/'preprocessing.json'),split_plan_sha256=sha256(BASE/'split_plan.json')))
    print('Pretraining check:',json.dumps(binary_rows),'UNSEEN:',json.dumps(unseen),flush=True)


def diagnostic_labels(partition):
    info=json.loads((METRICS/'pretraining_check.json').read_text())
    codebook=info['diagnostic_codebook']
    destination=CACHE/f'{partition}_diagnostics.npz'
    if destination.exists():
        with np.load(destination) as data:
            return data['y'],data['family'],codebook
    plan=json.loads((BASE/'split_plan.json').read_text())
    n=plan['partitions'][partition]['rows']
    y=np.empty(n,dtype=np.uint8); family=np.empty(n,dtype=np.uint8)
    mapping={name:i for i,name in enumerate(codebook)}
    offset=0
    for batch in pq.ParquetFile(plan['partitions'][partition]['membership_path']).iter_batches(columns=['target_binary','attack_label']):
        end=offset+len(batch)
        y[offset:end]=batch.column(0).to_numpy()
        family[offset:end]=[mapping[v] for v in batch.column(1).to_pylist()]
        offset=end
    require(offset==n,'Diagnostic label count mismatch')
    np.savez_compressed(destination,y=y,family=family)
    return y,family,codebook


def build_training_cache():
    preflight()
    destination=CACHE/'training_cache.json'
    if destination.exists():
        info=json.loads(destination.read_text())
        require(info['preprocessing_sha256']==sha256(BASE/'preprocessing.json'),'Cache/preprocessor mismatch')
        for key in ('x','y'):
            require(sha256(info[key+'_path'])==info[key+'_sha256'],'Training cache checksum mismatch')
        return
    plan=json.loads((BASE/'split_plan.json').read_text())
    state=json.loads((BASE/'preprocessing.json').read_text())
    n,d=plan['partitions']['train']['rows'],len(state['output_feature_names'])
    needed=n*d*4+n
    require(shutil.disk_usage(CACHE).free>needed+700_000_000,'Insufficient free disk for memory-mapped model-input cache')
    x_path=CACHE/'train_float32.dat'; y_path=CACHE/'train_y_uint8.dat'
    x=np.memmap(x_path,mode='w+',dtype='<f4',shape=(n,d))
    y=np.memmap(y_path,mode='w+',dtype='u1',shape=(n,))
    h64=hashlib.sha256(); hx=hashlib.sha256(); hy=hashlib.sha256()
    offset=0; started=time.perf_counter()
    for features,target,metadata in iter_transformed(prep_config(),'train',scaled=True):
        end=offset+len(features)
        original=np.ascontiguousarray(features,dtype='<f8')
        h64.update(original.tobytes())
        model_values=np.ascontiguousarray(features,dtype='<f4')
        require(np.isfinite(model_values).all(),'Model float32 conversion overflow')
        x[offset:end]=model_values; y[offset:end]=target
        hx.update(model_values.tobytes()); hy.update(np.asarray(target,dtype='u1').tobytes())
        offset=end
        if offset%1000000==0:
            x.flush();y.flush()
            print(f'Frozen training cache: {offset:,}/{n:,}',flush=True)
    require(offset==n,'Training cache is incomplete')
    with (BASE/'preprocessing_application_audit.csv').open(newline='') as f:
        expected=next(r['transformed_float64_sha256'] for r in csv.DictReader(f) if r['partition']=='train')
    require(h64.hexdigest()==expected,'Frozen transform differs from Phase 3; refusing model training')
    x.flush();y.flush();del x,y
    json_write(destination,dict(rows=n,features=d,x_path=str(x_path),y_path=str(y_path),
               x_sha256=hx.hexdigest(),y_sha256=hy.hexdigest(),preprocessing_sha256=sha256(BASE/'preprocessing.json'),
               frozen_float64_transform_sha256=h64.hexdigest(),dtype='float32',
               note='Model-input dtype conversion only; frozen float64 transform checksum exactly matches Phase 3.',
               creation_seconds=time.perf_counter()-started))
    for part in ('validation','test'):
        diagnostic_labels(part)
    print('Training cache complete; frozen float64 preprocessing reproduced exactly.',flush=True)


def training_arrays():
    guard()
    info=json.loads((CACHE/'training_cache.json').read_text())
    return (np.memmap(info['x_path'],mode='r',dtype='<f4',shape=(info['rows'],info['features'])),
            np.memmap(info['y_path'],mode='r',dtype='u1',shape=(info['rows'],)))


def batches(partition):
    for x,y,metadata in iter_transformed(prep_config(),partition,scaled=True):
        yield np.ascontiguousarray(x,dtype=np.float32),y


def class_weights():
    check=json.loads((METRICS/'pretraining_check.json').read_text())
    row=next(r for r in check['binary_counts'] if r['partition']=='train')
    return {0:row['records']/(2*row['benign']),1:row['records']/(2*row['malicious'])}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    preflight() if args.preflight_only else build_training_cache()
