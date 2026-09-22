"""Separate benign-only and label-aware locks; raw-score evaluation."""
import argparse,time
import joblib
import numpy as np
import torch
from .data import *
from .train import seed,Autoencoder,errors
from .metrics import benign_threshold,label_thresholds,metrics
from src.phase4.train import MemoryMonitor

LOCK_A=CONFIG/'benign_only_threshold_lock.json';LOCK_B=CONFIG/'label_aware_threshold_lock.json'

def names(): return [read(OUT/'selected_autoencoder.json')['name'],'isolation_forest']

def freeze_A():
    guard()
    if LOCK_A.exists():
        require(read(LOCK_A)['csv_sha256']==sha256(OUT/'benign_only_thresholds.csv'),'Protocol-A thresholds changed');return
    rows=[];artifacts={}
    for name in names():
        training=read(METRICS/f'{name}_training.json');path=METRICS/f'{name}_benign_validation.npy'
        require(sha256(path)==training['benign_validation_score_sha256'],'Benign validation scores changed')
        require(sha256(training['model_path'])==training['model_sha256'],'Model changed')
        score=np.load(path)
        for cap in PROTOCOL['protocol_A_caps']:
            rows.append(dict(model=name,protocol='A_benign_only',criterion=f'benign_fpr_at_most_{cap:g}',**benign_threshold(score,cap)))
        artifacts[name]=dict(model_path=training['model_path'],model_sha256=training['model_sha256'],benign_validation_score_sha256=sha256(path))
    write('benign_only_thresholds.csv',rows)
    json_write(LOCK_A,dict(frozen_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),malicious_validation_scores_evaluated=False,
        selection_uses_malicious_validation=False,selection_uses_test=False,selected_autoencoder_sha256=sha256(OUT/'selected_autoencoder.json'),
        rows=rows,artifacts=artifacts,csv_sha256=sha256(OUT/'benign_only_thresholds.csv')))
    print('PROTOCOL A LOCKED before malicious validation/test scoring',flush=True)

def score_model(name):
    guard();lock=read(LOCK_A);artifact=lock['artifacts'][name]
    require(sha256(artifact['model_path'])==artifact['model_sha256'],'Model changed after benign calibration')
    destination=METRICS/f'{name}_scoring.json'
    if destination.exists():
        r=read(destination);require(r['protocol_A_lock_sha256']==sha256(LOCK_A),'Protocol A changed')
        for part in r['partitions'].values(): require(sha256(part['score_path'])==part['score_sha256'],'Saved scores changed')
        return
    seed();training=read(METRICS/f'{name}_training.json')
    with MemoryMonitor() as memory:
        if name=='isolation_forest': model=joblib.load(artifact['model_path'])
        else:
            model=Autoencoder(training['config']['hidden_layers'],validation_benign().shape[1]);model.load_state_dict(torch.load(artifact['model_path'],weights_only=True,map_location='cpu'))
        records={}
        # Test scoring may precede B calibration; test labels are never read here.
        counts=read(BASE/'split_plan.json')['partitions']
        for part in ('train_benign','validation','test'):
            n=read(OUT/'anomaly_training_manifest.json')['benign_train_rows'] if part=='train_benign' else counts[part]['rows']
            dest=METRICS/f'{name}_{part}_scores.npy';scores=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float64,shape=(n,))
            batches=train_batches() if part=='train_benign' else partition_batches(part)
            start=time.perf_counter();compute=0.;offset=0;progress=0
            for x in batches:
                begin=time.perf_counter();values=-model.score_samples(x) if name=='isolation_forest' else errors(model,x);compute+=time.perf_counter()-begin
                require(np.isfinite(values).all(),'Nonfinite anomaly scores');scores[offset:offset+len(x)]=values;offset+=len(x)
                if offset-progress>=3_000_000: print(name,part,f'{offset:,}/{n:,} scored',flush=True);progress=offset
            require(offset==n,'Incomplete anomaly scoring');scores.flush();del scores
            elapsed=time.perf_counter()-start
            records[part]=dict(score_path=str(dest),score_sha256=sha256(dest),inference_seconds=elapsed,prediction_only_seconds=compute,
                latency_ms_per_1000=elapsed/n*1e6,evaluated_records=n)
            print('SCORED',name,part,round(elapsed,1),'seconds',flush=True)
        # Validate benign scores against the exact cache that underlies Protocol A.
        y,_,_=labels('validation');actual=np.load(records['validation']['score_path'],mmap_mode='r')[y==0]
        frozen=np.load(METRICS/f'{name}_benign_validation.npy')
        canonical=-model.score_samples(validation_benign()) if name=='isolation_forest' else errors(model,validation_benign())
        require(np.array_equal(canonical,frozen),'Same-layout benign calibration replay is not exact')
        # Float32 GEMM can choose a different reduction layout for final short
        # batches in the mixed partition. Exact same-layout replay above proves
        # the frozen checkpoint/cache is unchanged; bound the numerical drift.
        require(np.max(np.abs(actual-frozen)/(1+np.abs(frozen)))<1e-6,'Mixed-layout score drift exceeds float32 tolerance')
        # Preserve the exact benign calibration observations across batch shapes.
        val=np.load(records['validation']['score_path'],mmap_mode='r+')
        val[y==0]=frozen;val.flush();del val
        records['validation']['score_sha256']=sha256(records['validation']['score_path'])
        result=dict(model=name,protocol_A_lock_sha256=sha256(LOCK_A),model_sha256=artifact['model_sha256'],partitions=records,
            benign_replay_max_abs_difference=float(np.max(np.abs(actual-frozen))),same_layout_replay_exact=True,benign_calibration_scores_preserved_exactly=True)
    result['peak_memory_bytes']=memory.peak;json_write(destination,result)

def freeze_B():
    if LOCK_B.exists():
        require(read(LOCK_B)['csv_sha256']==sha256(OUT/'label_aware_thresholds.csv'),'Protocol-B thresholds changed');return
    lock=read(LOCK_A);rows=[];hashes={};y,_,_=labels('validation')
    for name in names():
        record=read(METRICS/f'{name}_scoring.json');require(record['protocol_A_lock_sha256']==sha256(LOCK_A),'Scores precede/violate A lock')
        path=record['partitions']['validation']['score_path'];require(sha256(path)==record['partitions']['validation']['score_sha256'],'Validation scores changed')
        scores=np.load(path);rows.extend(dict(model=name,protocol='B_label_aware',**r) for r in label_thresholds(y,scores));hashes[name]=sha256(path)
    write('label_aware_thresholds.csv',rows)
    json_write(LOCK_B,dict(frozen_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),selection_uses_test=False,
        validation_infilteration_is_used_for_calibration=True,rows=rows,validation_score_hashes=hashes,
        protocol_A_lock_sha256=sha256(LOCK_A),csv_sha256=sha256(OUT/'label_aware_thresholds.csv')))
    print('PROTOCOL B LOCKED using validation labels only',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['A','score','B'],required=True);p.add_argument('--name');a=p.parse_args()
    if a.stage=='A': freeze_A()
    elif a.stage=='B': freeze_B()
    else: score_model(a.name)
