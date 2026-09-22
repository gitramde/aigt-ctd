"""Exact columnwise distribution summaries without another full feature matrix."""
import json
import numpy as np
import pyarrow.parquet as pq
from .run import OUT,BASE,CACHE,PARTS,GROUPS,read,write,labels
from src.baseline.common import json_write


def ks_sorted(a,b):
    if not len(a) or not len(b): return None
    result=0.0
    # Check both sides of every jump of both empirical CDFs, including ties.
    for x in (a,b):
        jumps=x[np.r_[True,x[1:]!=x[:-1]]]
        for start in range(0,len(jumps),250000):
            values=jumps[start:start+250000]
            for side in ('left','right'):
                distance=np.abs(np.searchsorted(a,values,side=side)/len(a)-np.searchsorted(b,values,side=side)/len(b))
                result=max(result,float(distance.max(initial=0)))
    return result


def bins_from_reference(a):
    if not len(a): return np.array([-np.inf,np.inf])
    if a[0]==a[-1]: return np.array([-np.inf,np.nextafter(a[0],-np.inf),np.nextafter(a[0],np.inf),np.inf])
    return np.r_[-np.inf,np.unique(np.quantile(a,np.arange(.1,1,.1))),np.inf]


def psi(a,b,n_a,n_b):
    edges=bins_from_reference(a)
    pa=np.r_[np.histogram(a,bins=edges)[0],n_a-len(a)]/n_a
    pb=np.r_[np.histogram(b,bins=edges)[0],n_b-len(b)]/n_b
    pa=np.maximum(pa,1e-6);pb=np.maximum(pb,1e-6);pa/=pa.sum();pb/=pb.sum()
    return float(np.sum((pb-pa)*np.log(pb/pa)))


def stats(values):
    finite=values[np.isfinite(values)];finite.sort()
    result=dict(records=len(values),observed_records=len(finite),missing_rate=1-len(finite)/len(values))
    if len(finite):
        result.update(mean=float(np.mean(finite)),standard_deviation=float(np.std(finite,ddof=0)),median=float(np.median(finite)),
                      **{f'p{q}':float(np.percentile(finite,q)) for q in (10,25,75,90,99)})
    else: result.update({k:None for k in ('mean','standard_deviation','median','p10','p25','p75','p90','p99')})
    return finite,result


def read_column(plan,part,name,category=None):
    arrays=[]
    for entry in plan['partitions'][part]['files']:
        for batch in pq.ParquetFile(entry['cleaned_path']).iter_batches(columns=[name],batch_size=250000):
            if category is None: arrays.append(np.asarray(batch.column(0).to_numpy(zero_copy_only=False),dtype=np.float64))
            else:
                # Frozen category imputation; one-hot inputs are summarized as numerical 0/1 features.
                values=batch.column(0).to_pylist();mode=read(BASE/'preprocessing.json')['categorical_imputation_values'][name]
                arrays.append(np.fromiter((float((mode if v is None else v)==category) for v in values),dtype=float,count=len(values)))
    return np.concatenate(arrays)


def run():
    state=read(BASE/'preprocessing.json');plan=read(BASE/'split_plan.json')
    y=labels('train')[0];masks={}
    for part in ('validation','test'):
        _,codes,book=labels(part)
        masks[part+'_Infilteration']=codes==book.index('Infilteration')
        if part=='test': masks['test_Bot']=codes==book.index('Bot')
    specifications=[(c,c,None) for c in state['numeric_columns']]
    specifications += [(f'{c}={v}',c,v) for c in state['categorical_columns'] for v in state['categories'][c]]
    folder=OUT/'feature_checkpoints';folder.mkdir(exist_ok=True)
    overall=[];family=[];group_stats=[]
    pairs=[('training_benign',g) for g in ('validation_Infilteration','test_Infilteration','test_Bot')]
    pairs += [('training_malicious',g) for g in ('validation_Infilteration','test_Infilteration','test_Bot')]
    pairs += [('validation_Infilteration','test_Infilteration')]
    for index,(feature,column,category) in enumerate(specifications):
        checkpoint=folder/f'{index:03d}.json'
        if checkpoint.exists(): result=read(checkpoint)
        else:
            raw={p:read_column(plan,p,column,category) for p in PARTS}
            values=dict(raw)
            values['training_benign']=raw['train'][y==0];values['training_malicious']=raw['train'][y==1]
            for group,mask in masks.items(): values[group]=raw[group.split('_')[0]][mask]
            sorted_values={};summaries={}
            for group,array in values.items(): sorted_values[group],summaries[group]=stats(array)
            del raw,values
            def compare(reference,target):
                a=sorted_values[reference];b=sorted_values[target]
                return dict(feature=feature,feature_type='numeric' if category is None else 'frozen_one_hot',
                    reference_group=reference,target_group=target,**summaries[target],
                    **{'reference_'+k:v for k,v in summaries[reference].items()},
                    psi=0.0 if reference==target else psi(a,b,summaries[reference]['records'],summaries[target]['records']),
                    ks_statistic=0.0 if reference==target else ks_sorted(a,b),
                    representation='cleaned observed native units before imputation; fixed one-hot for Protocol',
                    psi_missing_bin=True,statistics_exact=True)
            result=dict(overall=[compare('train',p) for p in PARTS],family=[compare(a,b) for a,b in pairs],
                groups=[dict(feature=feature,group=g,**v) for g,v in summaries.items()])
            json_write(checkpoint,result)
        overall.extend(result['overall']);family.extend(result['family']);group_stats.extend(result['groups'])
        print(f'FEATURE SHIFT {index+1}/{len(specifications)}: {feature}',flush=True)
    write('feature_distribution_shift.csv',overall);write('family_feature_shift.csv',family);write('feature_group_statistics.csv',group_stats)
    top=[]
    for target in ('validation','test'):
        selected=sorted([r for r in overall if r['target_group']==target],key=lambda r:(r['ks_statistic'] or 0,r['psi']),reverse=True)[:20]
        top.extend(dict(rank=i+1,**r) for i,r in enumerate(selected))
    write('top20_feature_shifts.csv',top)
    both=[]
    for target in ('validation_Infilteration','test_Infilteration','test_Bot'):
        for feature,_,_ in specifications:
            rows=[r for r in family if r['feature']==feature and r['target_group']==target and r['reference_group'] in ('training_benign','training_malicious')]
            b=next(r for r in rows if r['reference_group']=='training_benign');m=next(r for r in rows if r['reference_group']=='training_malicious')
            both.append(dict(target_group=target,feature=feature,ks_vs_training_benign=b['ks_statistic'],ks_vs_training_malicious=m['ks_statistic'],
                psi_vs_training_benign=b['psi'],psi_vs_training_malicious=m['psi'],minimum_ks_against_both=min(b['ks_statistic'] or 0,m['ks_statistic'] or 0)))
    ranked=[]
    for target in ('validation_Infilteration','test_Infilteration','test_Bot'):
        ordered=sorted([r for r in both if r['target_group']==target],key=lambda r:r['minimum_ks_against_both'],reverse=True)
        ranked.extend(dict(rank=i+1,**r) for i,r in enumerate(ordered))
    write('family_shift_against_both_rankings.csv',ranked)
    print('FEATURE DISTRIBUTION ANALYSIS COMPLETE',flush=True)

if __name__=='__main__': run()
