"""Persist validation-selected operating points before held-out evaluation."""
import argparse,time
import numpy as np
from .data import *
from .train import seed,load,predict,path,MemoryMonitor
from src.phase4.metrics import binary_metrics
from src.phase4a.run import select_thresholds
BASELINES=('logistic_regression','random_forest','xgboost','mlp')
LOCK=CONFIG/'operating_threshold_lock.json'

def baseline_scores(name,part):
    info=read(BASE/'metrics'/f'{name}_seed42_evaluation.json')['partitions'][part]
    require(sha256(info['probability_path'])==info['probability_sha256'],'Baseline predictions changed')
    return np.load(info['probability_path'],mmap_mode='r')[cohort(part)['partition_index']]

def freeze():
    integrity()
    if LOCK.exists():
        lock=read(LOCK);require(lock['threshold_csv_sha256']==sha256(OUT/'thresholds.csv'),'Threshold artifact changed');return
    selected=read(OUT/'selected_model.json');names=[selected['config']['name'],'transformer_L1_control']
    c=cohort('validation');y=c['target_binary'];rows=[];artifacts={}
    for name in names+list(BASELINES):
        if name in names:
            record=read(METRICS/f'{name}_training.json');probability_path=METRICS/f'{name}_validation.npy'
            require(record['model_sha256']==sha256(path(name)) and record['validation_prediction_sha256']==sha256(probability_path),'Selected artifact changed')
            p=np.load(probability_path);artifacts[name]=dict(model_sha256=record['model_sha256'],validation_sha256=sha256(probability_path))
        else: p=baseline_scores(name,'validation')
        require(len(p)==len(y),'Matched validation size mismatch')
        candidates=[dict(criterion='fixed_0_5',threshold=.5)]+select_thresholds(y,p)
        allowed={'fixed_0_5','maximum_macro_f1','fpr_at_most_0.01','fpr_at_most_0.005','fpr_at_most_0.001'} if name in names else {'fixed_0_5','fpr_at_most_0.01'}
        for candidate in candidates:
            if candidate['criterion'] not in allowed: continue
            metrics=binary_metrics(y,p,candidate['threshold'],include_auc=False)
            rows.append(dict(model=name,criterion=candidate['criterion'],threshold=candidate['threshold'],selection_partition='validation',
                **{'validation_'+k:v for k,v in metrics.items()}))
    write('thresholds.csv',rows)
    json_write(LOCK,dict(frozen_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),thresholds=rows,
        threshold_csv_sha256=sha256(OUT/'thresholds.csv'),artifacts=artifacts,selection_uses_test=False,
        selected_model_sha256=sha256(OUT/'selected_model.json'),cohort_manifest_sha256=sha256(METRICS/'sequence_manifest.json'),
        baseline_policy='reselect FPR <= 1% using saved baseline predictions on the same matched validation targets; no baseline refitting',
        tie_policy='maximize validation macro-F1; FPR caps maximize recall then minimize false positives then prefer higher threshold'))
    print('OPERATING THRESHOLDS LOCKED BEFORE TEST',flush=True)

def evaluate(name):
    integrity();lock=read(LOCK);require(sha256(OUT/'thresholds.csv')==lock['threshold_csv_sha256'],'Threshold lock mismatch')
    require(sha256(OUT/'selected_model.json')==lock['selected_model_sha256'],'Model selection changed')
    require(sha256(METRICS/'sequence_manifest.json')==lock['cohort_manifest_sha256'],'Cohorts changed')
    require(sha256(path(name))==lock['artifacts'][name]['model_sha256'],'Model changed after freeze')
    output=METRICS/f'{name}_evaluation.json'
    if output.exists():
        prior=read(output)
        require(prior['model_sha256']==sha256(path(name)) and prior['threshold_lock_sha256']==sha256(LOCK),'Evaluation provenance mismatch')
        for r in prior['partitions'].values(): require(sha256(r['prediction_path'])==r['prediction_sha256'],'Saved prediction changed')
        print('EVALUATION ALREADY COMPLETE',name,flush=True);return
    seed();config=read(CONFIG/f'{name}.json');result=dict(model=name,config=config,seed=42,model_sha256=sha256(path(name)),threshold_lock_sha256=sha256(LOCK),partitions={})
    book=read(METRICS/'sequence_manifest.json')['codebook'];train_labels=read(BASE/'split_plan.json')['partitions']['train']['class_counts']
    with MemoryMonitor() as memory:
        model=load(config)
        for part in ('validation','test'):
            begin=time.perf_counter();x=features(part);load_seconds=time.perf_counter()-begin;c=cohort(part)
            p,timing=predict(model,x,c['partition_index'],config['length']);del x
            dest=METRICS/f'{name}_{part}_evaluation.npy';np.save(dest,p)
            at_half=binary_metrics(c['target_binary'],p,.5)
            if part=='validation':
                train_record=read(METRICS/f'{name}_training.json')
                require(abs(at_half['macro_f1']-train_record['selected_validation_macro_f1'])<1e-12,'Reloaded checkpoint differs from selection')
            metrics=[];family=[]
            for t in lock['thresholds']:
                if t['model']!=name: continue
                m=binary_metrics(c['target_binary'],p,t['threshold'],include_auc=False)
                metrics.append(dict(model=name,partition=part,seed=42,criterion=t['criterion'],**m,
                    **{k:at_half[k] for k in ('roc_auc','pr_auc','average_precision')}))
                for code,label in enumerate(book):
                    if label=='Benign': continue
                    mask=c['family_code']==code;n=int(mask.sum());values=p[mask];detected=int((values>=t['threshold']).sum())
                    family.append(dict(model=name,partition=part,criterion=t['criterion'],threshold=t['threshold'],attack_family=label,
                        training_status='KNOWN' if train_labels.get(label,0)>0 else 'UNSEEN',N=n,detected=detected,missed=n-detected,
                        recall=detected/n if n else None,mean_score=float(values.mean()) if n else None,
                        median_score=float(np.median(values)) if n else None,
                        **{f'p{q}':float(np.percentile(values,q)) if n else None for q in (90,95,99)}))
            result['partitions'][part]=dict(metrics=metrics,families=family,runtime=dict(**timing,feature_load_seconds=load_seconds,
                end_to_end_inference_seconds=load_seconds+timing['inference_seconds'],end_to_end_latency_ms_per_1000=(load_seconds+timing['inference_seconds'])/len(p)*1e6),
                prediction_path=str(dest),prediction_sha256=sha256(dest))
            print('EVALUATED',name,part,'F1',at_half['f1'],'recall',at_half['recall'],flush=True)
    result['peak_memory_bytes']=memory.peak;json_write(output,result);integrity()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--model');a=p.parse_args()
    freeze() if a.freeze else evaluate(a.model)
