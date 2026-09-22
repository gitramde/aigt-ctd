"""Freeze every validation operating point before any test inference."""
import argparse,time
import joblib
import numpy as np
import torch
import xgboost as xgb
from .common import *
from .data import GraphData
from .train import seed,make_model,predict
from .baselines import NAMES,predict as baseline_predict
from src.phase4a.run import select_thresholds
from src.phase4.metrics import binary_metrics
from src.phase4.train import MemoryMonitor

LOCK=CONFIG/'threshold_lock.json'

def freeze():
    guard();selection=read(OUT/'selected_model.json');names=selection['evaluation_models']
    data=GraphData();y=data.y[data.part==1];rows=[];artifacts={}
    if LOCK.exists():
        lock=read(LOCK);require(lock['selected_model_sha256']==sha256(OUT/'selected_model.json'),'Selection changed')
        require(lock['thresholds_sha256']==sha256(OUT/'thresholds.csv'),'Threshold table changed');return
    for name in names:
        record=read(METRICS/f'{name}_training.json');path=METRICS/f'{name}_validation.npy'
        require(sha256(path)==record['validation_prediction_sha256'],'Validation predictions changed')
        p=np.load(path);require(len(p)==len(y),'Validation cohort mismatch')
        points=[dict(criterion='fixed_0_5',threshold=.5)]+[r for r in select_thresholds(y,p) if r['criterion'] in ('maximum_macro_f1','fpr_at_most_0.01')]
        rows.extend(dict(model=name,criterion=r['criterion'],threshold=r['threshold'],selection_partition='validation',
            **{'validation_'+k:v for k,v in binary_metrics(y,p,r['threshold'],include_auc=False).items()}) for r in points)
        model_path=record.get('model_path',str(MODELS/f'{name}.pt'))
        require(sha256(model_path)==record['model_sha256'],'Selected model changed')
        artifacts[name]=dict(model_path=model_path,model_sha256=record['model_sha256'],validation_prediction_sha256=sha256(path),training_record_sha256=sha256(METRICS/f'{name}_training.json'))
    write('thresholds.csv',rows)
    json_write(LOCK,dict(frozen_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),selection_uses_test=False,
        selected_model_sha256=sha256(OUT/'selected_model.json'),cohort_sha256=sha256(METRICS/'evaluation_cohort.parquet'),
        thresholds_sha256=sha256(OUT/'thresholds.csv'),thresholds=rows,artifacts=artifacts,
        tie_policy='macro-F1 ties choose higher threshold; FPR cap maximizes recall then minimizes FP then higher threshold'))
    print('ALL OPERATING POINTS LOCKED BEFORE TEST',flush=True)


def evaluate(name):
    guard();lock=read(LOCK);artifact=lock['artifacts'][name]
    require(sha256(OUT/'selected_model.json')==lock['selected_model_sha256'],'Selection mismatch')
    require(sha256(METRICS/'evaluation_cohort.parquet')==lock['cohort_sha256'],'Cohort mismatch')
    require(sha256(artifact['model_path'])==artifact['model_sha256'],'Model mismatch')
    require(sha256(METRICS/f'{name}_training.json')==artifact['training_record_sha256'],'Training provenance mismatch')
    destination=METRICS/f'{name}_evaluation.json'
    if destination.exists():
        old=read(destination);require(old['threshold_lock_sha256']==sha256(LOCK),'Existing evaluation lock mismatch')
        for entry in old['partitions'].values(): require(sha256(entry['prediction_path'])==entry['prediction_sha256'],'Existing predictions changed')
        return
    record=read(METRICS/f'{name}_training.json');seed()
    with MemoryMonitor() as memory:
        data=GraphData()
        if name in NAMES:
            if name=='xgboost': model=xgb.Booster();model.load_model(artifact['model_path'])
            else: model=joblib.load(artifact['model_path'])
        else:
            model=make_model(record['config']);model.load_state_dict(torch.load(artifact['model_path'],weights_only=True,map_location='cpu'))
        output=dict(model=name,threshold_lock_sha256=sha256(LOCK),model_sha256=artifact['model_sha256'],partitions={})
        for part in ('validation','test'):
            mask=data.part==PARTS.index(part);y=data.y[mask]
            if name in NAMES:
                start=time.perf_counter();p=baseline_predict(model,name,np.asarray(data.x[mask]));elapsed=time.perf_counter()-start
                timing=dict(inference_seconds=elapsed,graph_preparation_seconds=0.,prediction_only_seconds=elapsed,latency_ms_per_1000=elapsed/len(y)*1e6,evaluated_edges=len(y))
            else: p,timing=predict(model,data,part,record['config']['window_minutes'])
            replay_difference=0.
            if part=='validation':
                frozen=np.load(METRICS/f'{name}_validation.npy')
                require(np.allclose(p,frozen,rtol=1e-6,atol=1e-7),'Reloaded validation mismatch')
                replay_difference=float(np.max(np.abs(p-frozen)))
                # Preserve exact threshold ties from the locked validation scores;
                # parallel tree reductions can differ in their final floating bit.
                p=frozen
            path=METRICS/f'{name}_{part}_evaluation.npy';np.save(path,p);ranking=binary_metrics(y,p);rows=[]
            for threshold in lock['thresholds']:
                if threshold['model']!=name: continue
                rows.append(dict(model=name,partition=part,criterion=threshold['criterion'],seed=42,
                    **binary_metrics(y,p,threshold['threshold'],include_auc=False),
                    **{k:ranking[k] for k in ('roc_auc','pr_auc','average_precision')}))
            output['partitions'][part]=dict(metrics=rows,runtime=timing,prediction_path=str(path),prediction_sha256=sha256(path),validation_replay_max_abs_difference=replay_difference)
            print('EVALUATED',name,part,'macro-F1',ranking['macro_f1'],'AP',ranking['average_precision'],flush=True)
    output['peak_memory_bytes']=memory.peak;json_write(destination,output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--name');a=p.parse_args()
    freeze() if a.freeze else evaluate(a.name)
