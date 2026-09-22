"""Recompute each saved score table and derive only frozen comparisons."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from .common import *
from .evaluate import cohort,rows_for
from .metrics import thresholds,ranking,counts

def main():
    check_lock();allrows=[];families=[];runtime=[];overlap=[]
    for seed in SEEDS:
        for group,names in GROUPS.items():
            for name in names:
                p=folder(seed,group);record=read(p/'metrics'/f'{name}_evaluation.json')
                for artifact in record['artifacts']:require(sha(ROOT/artifact['path'])==artifact['sha256'],'Final score/ID artifact changed')
                lock=read(p/'configs'/f'{name}_threshold_lock.json')
                checkpoint=p/'models'/(name+('.ubj' if name=='xgboost' else '.joblib' if group=='supervised' else '.pt'))
                require(sha(checkpoint)==record['checkpoint_sha256']==lock['checkpoint_sha256'],'Final checkpoint provenance mismatch')
                require(sha(p/'configs'/f'{name}_threshold_lock.json')==record['threshold_lock_sha256'],'Threshold lock changed')
                original=pd.read_csv(p/'metrics'/f'{name}_metrics.csv')
                for part in ('validation','test'):
                    score=np.load(p/'metrics'/f'{name}_{part}_scores.npy');y,f,book,ids,cn=cohort(group,part)
                    require(np.array_equal(ids,np.load(p/'metrics'/f'{name}_{part}_ids.npy')),'Final cohort equality failed')
                    rows,fr,_=rows_for(seed,group,name,part,score,lock['thresholds'])
                    for r in rows:
                        pred=score>=r['threshold'];tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
                        require([tn,fp,fn,tp]==[r[k] for k in ('TN','FP','FN','TP')],'Independent confusion recomputation mismatch')
                        reference=dict(accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),macro_f1=f1_score(y,pred,average='macro',zero_division=0))
                        saved=original[(original.partition==part)&(original.criterion==r['criterion'])].iloc[0]
                        for key,value in reference.items():require(np.isclose(value,r[key],rtol=0,atol=1e-12) and np.isclose(saved[key],value,rtol=0,atol=1e-12),'Independent metric mismatch: '+key)
                    allrows+=rows;families+=fr
                training=read(p/'metrics'/f'{name}_training.json')
                size=checkpoint.stat().st_size;pipeline=size+(folder(seed,'graph')/'models/gatv2.pt').stat().st_size if group=='graph_temporal' else size
                for t in record['timings']:runtime.append(dict(seed=seed,model=name,group=group,training_seconds=training['training_seconds'],peak_memory_bytes=max(training['peak_memory_bytes'],record['peak_memory_bytes']),model_size_bytes=size,total_required_model_bytes=pipeline,**t))
        # Matched conventional controls are sliced from this seed's full-cohort predictions.
        for name in GROUPS['supervised']:
            p=folder(seed,'supervised');vy,vf,book,vids,cn=cohort('temporal','validation')
            locked=read(p/'configs'/f'{name}_matched_threshold_lock.json')['thresholds']
            dest=OUT/f'seed_{seed}/matched_controls';dest.mkdir(exist_ok=True)
            write(dest/f'{name}_threshold_lock.json',dict(seed=seed,thresholds=locked,validation_ids_sha256=sha(ROOT/'results/temporal_v1/metrics/validation_cohort.parquet'),derived_from_same_seed_full_predictions=True))
            for part in ('validation','test'):
                y,f,book,ids,cn=cohort('temporal',part);score=np.load(p/'metrics'/f'{name}_{part}_scores.npy')[ids]
                np.save(dest/f'{name}_{part}_scores.npy',score)
                rows,fr,_=rows_for(seed,'temporal',name,part,score,locked);allrows+=rows;families+=fr
        rf=folder(seed,'supervised');ae=folder(seed,'anomaly')
        rt=next(r['threshold'] for r in read(rf/'configs/random_forest_threshold_lock.json')['thresholds'] if r['criterion']=='fpr_at_most_0.01')
        at=next(r['threshold'] for r in read(ae/'configs/autoencoder_B_threshold_lock.json')['thresholds'] if r['criterion']=='benign_fpr_at_most_0.01')
        for part in ('validation','test'):
            y,f,book,ids,cn=cohort('supervised',part);r=np.load(rf/f'metrics/random_forest_{part}_scores.npy')>=rt;a=np.load(ae/f'metrics/autoencoder_B_{part}_scores.npy')>=at
            score=(r|a).astype(float);base=dict(seed=seed,model='rf_or_ae',cohort=cn,partition=part,criterion='component_1_percent',threshold=.5)
            allrows.append(dict(**base,**counts(y,score,.5),**ranking(y,score),ranking_basis='binary_decision'))
            for i,label in enumerate(book):
                if label=='Benign':continue
                mask=f==i;n=int(mask.sum());hits=int((mask&(r|a)).sum());both=int((mask&r&a).sum());union=int((mask&(r|a)).sum())
                families.append(dict(**base,attack_family=label,N=n,detected=hits,recall=hits/n if n else None))
                overlap.append(dict(seed=seed,partition=part,attack_family=label,N=n,rf_only=int((mask&r&~a).sum()),ae_only=int((mask&a&~r).sum()),both=both,neither=int((mask&~r&~a).sum()),jaccard=both/union if union else None))
    df=pd.DataFrame(allrows);ff=pd.DataFrame(families)
    dest=OUT/'aggregate';dest.mkdir(exist_ok=True)
    df.to_csv(dest/'per_seed_metrics.csv',index=False);ff.to_csv(dest/'per_seed_family_metrics.csv',index=False)
    pd.DataFrame(runtime).to_csv(dest/'runtime.csv',index=False);pd.DataFrame(overlap).to_csv(dest/'rf_ae_overlap.csv',index=False)
    comparison=df[(df.cohort=='full_dataset') & (df.partition=='test') & (((df.model=='random_forest') & (df.criterion=='fpr_at_most_0.01')) | ((df.model=='autoencoder_B') & (df.criterion=='benign_fpr_at_most_0.01')) | (df.model=='rf_or_ae'))].copy()
    for family in ('Bot','Infilteration'):
        temp=ff[(ff.attack_family==family)&(ff.cohort=='full_dataset')&(ff.partition=='test')][['seed','model','criterion','recall']].rename(columns={'recall':family+'_recall'})
        comparison=comparison.merge(temp,on=['seed','model','criterion'],validate='one_to_one')
    comparison.to_csv(dest/'rf_ae_or_comparison.csv',index=False)
    keys=['model','cohort','partition','criterion'];metrics=SPEC['statistics']['metrics']+['false_negative_rate']
    grouped=df.groupby(keys,dropna=False);require((grouped.seed.nunique()==5).all(),'Missing required seed in aggregate')
    summary=grouped[metrics].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary['completed_seeds']=5;summary.to_csv(dest/'five_seed_mean_sample_sd.csv')
    family_summary=ff.groupby(keys+['attack_family'],dropna=False).recall.agg(['mean','std','count']);family_summary.to_csv(dest/'family_mean_sample_sd.csv')
    deltas=[]
    for treated,control,cn in [('gatv2','edge_mlp','feb20_matched'),('gatv2','gatv2_self_only','feb20_matched'),('transformer_L64','transformer_L1','phase5_matched'),('gat_transformer_L8','gat_transformer_L1','feb20_matched')]:
        a=df[(df.model==treated)&(df.cohort==cn)];b=df[(df.model==control)&(df.cohort==cn)]
        merged=a.merge(b,on=['seed','cohort','partition','criterion'],suffixes=('_a','_b'),validate='one_to_one')
        for _,r in merged.iterrows():
            base={k:r[k] for k in ('seed','cohort','partition','criterion')};base.update(model=treated,control=control)
            for metric in ('recall','precision','f1','macro_f1','false_positive_rate','roc_auc','average_precision'):base['delta_'+metric]=r[metric+'_a']-r[metric+'_b']
            deltas.append(base)
    dd=pd.DataFrame(deltas);dd.to_csv(dest/'paired_per_seed_differences.csv',index=False)
    ds=dd.groupby(['model','control','cohort','partition','criterion'])[[c for c in dd if c.startswith('delta_')]].agg(['mean','std']);ds.to_csv(dest/'paired_mean_sample_sd.csv')
    write(OUT/'audit/final_verification.json',dict(status='PASS',completed_required_fits=55,completed_self_only_fits=5,independent_confusion_and_metric_recomputation=True,cohort_equality=True,completed_seeds=SEEDS))
    (OUT/'FINAL_EXECUTION_SUMMARY.md').write_text('# Phase 11 final execution complete\n\nAll 55 mandatory fits and 5 planned self-only fits completed freshly across seeds 42, 123, 456, 789, 1024. No development fit was substituted.\n\n'
        'Saved scores, cohort IDs, threshold locks and metrics passed independent verification. Per-seed metrics, five-seed arithmetic means and sample SD, paired differences, family recalls, RF/AE/OR overlaps and runtime measurements are in aggregate/.\n\n'
        'These fixed-split results measure training stochasticity. They are not a new confirmatory holdout or independent dataset replications. Runtime remains CPU-specific and must be compared within the recorded cohort/stage scopes. No further experimental phase was started.\n',encoding='utf-8')

if __name__=='__main__':main()
