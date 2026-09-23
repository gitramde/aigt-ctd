"""Independent score-based verification and separate Phase11B reports."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve, average_precision_score, auc)
from .phase11b import (AUDIT, ORDER, ROOT, OUT, SPEC, SEEDS, read, write, sha, require,
    folder, selection_seed, dependency, verify_sources, protected, environment)
from .evaluate import cohort
from .metrics import thresholds

DEST=OUT/'aggregate_phase11b'
METRICS=SPEC['statistics']['metrics']+['false_negative_rate']
DELTA=['recall','precision','f1','macro_f1','false_positive_rate','roc_auc','average_precision']

def independent(y,score,threshold):
    pred=score>=threshold
    tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    precision,recall,_=precision_recall_curve(y,score)
    return dict(N=len(y),TN=tn,FP=fp,FN=fn,TP=tp,accuracy=accuracy_score(y,pred),
        precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred,zero_division=0),
        f1=f1_score(y,pred,zero_division=0),macro_f1=f1_score(y,pred,labels=[0,1],average='macro',zero_division=0),
        false_positive_rate=fp/(fp+tn) if fp+tn else 0.,false_negative_rate=fn/(fn+tp) if fn+tp else 0.,
        roc_auc=roc_auc_score(y,score),pr_auc=auc(recall,precision),average_precision=average_precision_score(y,score))

def aggregate(frame,keys,columns):
    counts=frame.groupby(keys).seed.nunique()
    require((counts==5).all(),'Missing seeds; no partial five-seed aggregation')
    result=frame.groupby(keys)[columns].agg(['mean','std'])
    result.columns=['_'.join(c) for c in result.columns]
    result['completed_seeds']=counts
    return result.reset_index()

def main():
    verify_sources();environment()
    require(not DEST.exists(),'Refusing to overwrite Phase11B aggregate')
    rows=[];runtime=[];manifest=[];families=[]
    for seed in SEEDS:
        encoder=dependency(seed)
        for group,name in ORDER:
            p=folder(seed,group);train=read(p/'metrics'/f'{name}_training.json')
            evaluation=read(p/'metrics'/f'{name}_evaluation.json')
            lockpath=p/'configs'/f'{name}_threshold_lock.json';lock=read(lockpath)
            checkpoint=p/'models'/f'{name}.pt'
            require(selection_seed(train)==seed and evaluation['seed']==lock['seed']==seed,'Seed mismatch')
            require(train['status']==evaluation['status']=='complete','Incomplete fit/evaluation')
            require(sha(checkpoint)==train['model_sha256']==evaluation['checkpoint_sha256']==lock['checkpoint_sha256'],'Checkpoint hash mismatch')
            require(sha(lockpath)==evaluation['threshold_lock_sha256'],'Threshold lock changed')
            require(lock['frozen_before_test'] and lock['spec_sha256']==sha(ROOT/'results/final_spec_v1/final_experiment_spec.json'),'Threshold provenance')
            for artifact in evaluation['artifacts']:
                require(sha(ROOT/artifact['path'])==artifact['sha256'],'Score/ID artifact changed')
            validation=np.load(p/'metrics'/f'{name}_validation_scores.npy')
            vy,_,_,vids,_=cohort(group,'validation')
            require(sha(p/'metrics'/f'{name}_validation_scores.npy')==lock['validation_scores_sha256'],'Calibration scores changed')
            require(hashlib.sha256(np.asarray(vids,dtype='<i8').tobytes()).hexdigest()==lock['cohort_ids_sha256'],'Calibration cohort changed')
            require(thresholds(vy,validation,group)==lock['thresholds'],'Recalibrated thresholds differ')
            saved=pd.read_csv(p/'metrics'/f'{name}_metrics.csv')
            for part in ('validation','test'):
                y,_,_,ids,cn=cohort(group,part)
                score=np.load(p/'metrics'/f'{name}_{part}_scores.npy')
                require(np.array_equal(ids,np.load(p/'metrics'/f'{name}_{part}_ids.npy')),'Exact cohort/order mismatch')
                require(len(score)==len(y) and np.isfinite(score).all(),'Invalid scores')
                for point in lock['thresholds']:
                    values=independent(y,score,point['threshold'])
                    reference=saved[(saved.partition==part)&(saved.criterion==point['criterion'])]
                    require(len(reference)==1,'Duplicate/missing saved metric')
                    for key,value in values.items():
                        require(np.isclose(value,reference.iloc[0][key],rtol=0,atol=1e-12),'Independent metric mismatch: '+key)
                    base=dict(seed=seed,model=name,cohort=cn,partition=part,**point)
                    rows.append(dict(**base,**values))
                    families.append(dict(**base,attack_family='DDoS attacks-LOIC-HTTP',N=int(y.sum()),
                        detected=values['TP'],recall=values['recall']))
            fit_runtime=read(AUDIT/f'fit_{seed}_{name}_runtime.json')
            eval_runtime=read(AUDIT/f'evaluate_{seed}_{name}_runtime.json')
            representation=None
            if group=='graph_temporal':
                representation=read(p/'metrics/representation_manifest.json')
                require(representation['encoder_sha256']==encoder,'Wrong shared encoder')
                require(sha(p/'metrics/representation_manifest.json')==train['representation_manifest_sha256'],'Representation provenance changed')
            peak=max(fit_runtime['process_peak_memory_bytes'],eval_runtime['process_peak_memory_bytes'],
                train['peak_memory_bytes'],evaluation['peak_memory_bytes'],
                representation['peak_memory_bytes'] if representation else 0)
            for timing in evaluation['timings']:
                n=88782 if timing['partition']=='validation' else 212591
                r=dict(seed=seed,model=name,cohort='feb20_matched',**timing,
                    fit_stage_seconds=fit_runtime['stage_seconds'],training_seconds=train['training_seconds'],
                    validation_selection_seconds=train['validation_selection_seconds'],
                    training_input_seconds=fit_runtime['training_input_seconds'],
                    fitting_excluding_measured_graph_input_seconds=train['training_seconds']-fit_runtime['training_input_seconds'],
                    training_metadata_loading_seconds=fit_runtime['metadata_loading_seconds'],
                    peak_memory_bytes=peak,model_size_bytes=checkpoint.stat().st_size,
                    pipeline_size_bytes=checkpoint.stat().st_size+((folder(seed,'graph')/'models/gatv2.pt').stat().st_size if representation else 0),
                    timing_scope=fit_runtime['scope'],repeated_training_cost='One fit cost repeated across partition rows; do not sum')
                r['latency_ms_per_1000']=r['inference_seconds']*1000000/n
                for key,value in r.items():
                    if key.endswith('_seconds'):require(np.isfinite(value) and value>=0,'Invalid runtime: '+key)
                runtime.append(r)
            manifest.append(dict(seed=seed,model=name,group=group,status='complete',checkpoint_sha256=sha(checkpoint),
                threshold_lock_sha256=sha(lockpath),training_record_sha256=sha(p/'metrics'/f'{name}_training.json'),
                evaluation_record_sha256=sha(p/'metrics'/f'{name}_evaluation.json'),
                effective_configuration_sha256=sha(OUT/f'audit/effective/seed_{seed}/{name}.json'),
                encoder_sha256=encoder if representation else '',selected_step=train.get('selected_epoch'),
                environment_sha256=sha(AUDIT/f'fit_{seed}_{name}_environment.json')))
    df=pd.DataFrame(rows);keys=['model','cohort','partition','criterion']
    summary=aggregate(df,keys,METRICS)
    deltas=[]
    for model,control in [('gatv2','edge_mlp'),('gatv2','gatv2_self_only'),('gat_transformer_L8','gat_transformer_L1')]:
        a=df[df.model==model];b=df[df.model==control]
        merged=a.merge(b,on=['seed','cohort','partition','criterion'],suffixes=('_a','_b'),validate='one_to_one')
        require(len(merged)==len(a)==len(b),'Incomplete paired comparison')
        for _,r in merged.iterrows():
            deltas.append(dict(seed=r.seed,model=model,control=control,cohort=r.cohort,partition=r.partition,criterion=r.criterion,
                **{'delta_'+m:r[m+'_a']-r[m+'_b'] for m in DELTA}))
    dd=pd.DataFrame(deltas)
    ds=aggregate(dd,['model','control','cohort','partition','criterion'],['delta_'+m for m in DELTA])
    checks=protected(recheck=True)
    DEST.mkdir()
    for name,frame in [('per_seed_metrics',df),('aggregate_metrics',summary),('paired_component_differences',dd),
        ('paired_aggregate_differences',ds),('runtime_metrics',pd.DataFrame(runtime)),('run_manifest',pd.DataFrame(manifest)),
        ('per_seed_family_metrics',pd.DataFrame(families)),('protected_artifact_verification',pd.DataFrame(checks))]:
        frame.to_csv(DEST/f'{name}.csv',index=False)
    shared=[]
    for seed in SEEDS:
        record=read(folder(seed,'graph_temporal')/'metrics/representation_manifest.json')
        shared.extend(dict(seed=seed,stage='same_seed_shared_representation',peak_memory_bytes=record['peak_memory_bytes'],**r) for r in record['timings'])
    history=read(AUDIT/'history_replay.json')
    shared.extend(dict(seed='shared_all_seeds',stage='history_replay_including_diagnostic_counts',peak_memory_bytes=history['peak_memory_bytes'],**r) for r in history['timings'])
    pd.DataFrame(shared).to_csv(DEST/'shared_preparation_runtime.csv',index=False)
    text=['# Phase 11B execution summary','','Status: **COMPLETE**. 20 mandatory and 5 self-only fits completed.','',
        'All five seeds were fitted freshly. Same-seed frozen GAT encoders supplied both temporal heads. No development fits substituted.',
        '', 'Independent metric/threshold/cohort/checkpoint verification passed. Protected artifacts unchanged.',
        '', 'Mean ± sample SD (ddof=1), test metrics; five completed seeds. Positive paired FPR differences mean more false alarms.','']
    for model,control in [('gatv2','edge_mlp'),('gatv2','gatv2_self_only'),('gat_transformer_L8','gat_transformer_L1')]:
        text += [f'## {model} versus {control}','','| Operating point | Metric | Model mean ± SD | Control mean ± SD | Paired difference mean ± SD |','| --- | --- | --- | --- | --- |']
        for _,r in ds[(ds.model==model)&(ds.control==control)&(ds.partition=='test')].iterrows():
            a=summary[(summary.model==model)&(summary.partition=='test')&(summary.criterion==r.criterion)].iloc[0]
            b=summary[(summary.model==control)&(summary.partition=='test')&(summary.criterion==r.criterion)].iloc[0]
            def fmt(v):return f'{v:.6e}' if 0<abs(v)<1e-6 else f'{v:.6f}'
            for m in DELTA:
                text.append(f'| {r.criterion} | {m} | {fmt(a[m+"_mean"])} ± {fmt(a[m+"_std"])} | {fmt(b[m+"_mean"])} ± {fmt(b[m+"_std"])} | {fmt(r["delta_"+m+"_mean"])} ± {fmt(r["delta_"+m+"_std"])} |')
    text += ['', 'Individual seed results and all metrics are in per_seed_metrics.csv. No contribution is established solely by one favorable seed or operating point.',
        '', 'Seed variation measures training stochasticity on the fixed, previously observed Feb20 split. No significance tests or independent dataset replications are claimed.',
        '', 'Runtime: graph input construction is measured separately; fit residuals still include minibatch overhead. Head inference includes lookup. Shared encoder extraction is counted once per seed; history replay includes diagnostic counts and is shared across seeds. Raw cleaning/preprocessing/cache creation was not rerun; timings are not raw-data end-to-end serving latency. Pipeline memory is the maximum of separate measured stages, not their sum.',
        '', 'The initial restricted-process failure and successful existing-overlay recovery remain preserved. No packages were installed or changed during recovery. Environment checks ran inside every fit process.',
        '', 'STOP: no XAI, tuning, model changes, fusion changes, or paper rewriting was started.']
    (DEST/'PHASE_11B_EXECUTION_SUMMARY.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    (DEST/'integrity_report.md').write_text('# Integrity: PASS\n\nAll required seeds, saved scores, exact cohort IDs, thresholds, checkpoint hashes, same-seed encoder dependencies, runtime records and protected artifacts verified. Historical hash limitations of the recovery baseline remain explicit in its unchanged audit.\n',encoding='utf-8')
    files=list(DEST.glob('*'))+list(AUDIT.rglob('*'))
    for seed in SEEDS:
        for group in ('graph','graph_temporal'): files+=list(folder(seed,group).rglob('*'))
    pd.DataFrame([dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(files)) if p.is_file()]).to_csv(DEST/'artifact_hash_manifest.csv',index=False)
    previous=OUT/'PHASE_11B_EXECUTION_SUMMARY.md'
    if previous.exists():
        import shutil
        shutil.copyfile(previous,AUDIT/'previous_execution_summary.md')
    previous.write_text((DEST/'PHASE_11B_EXECUTION_SUMMARY.md').read_text(encoding='utf-8'),encoding='utf-8')
    write(AUDIT/'complete.json',dict(status='PASS',mandatory=20,self_only=5,training_complete=True))
    print('Phase11B COMPLETE; STOP.',flush=True)

if __name__=='__main__':main()
