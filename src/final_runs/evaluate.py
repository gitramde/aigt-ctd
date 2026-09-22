"""Separate final evaluation worker: validation locks precede test scoring."""
import argparse,time
import numpy as np
from .common import *
from .metrics import thresholds,ranking,counts

def cohort(group,part):
    if group in ('supervised','anomaly'):
        from src.phase4.data import diagnostic_labels
        y,f,book=diagnostic_labels(part)
        return y,f,book,np.arange(len(y),dtype=np.int64),'full_dataset'
    if group=='temporal':
        from src.temporal.data import cohort as get
        c=get(part);book=SPEC['data']['full_temporal_cohort']['codebook']
        return c['target_binary'],c['family_code'],book,c['partition_index'],'phase5_matched'
    import pyarrow.parquet as pq
    meta=pq.read_table(ROOT/'data/graph_v1/metadata.parquet',columns=['target_binary','partition']).to_pandas()
    ids=np.load(ROOT/'data/graph_v1/target_ids.npy');mask=meta.partition.to_numpy()[ids]==('train','validation','test').index(part)
    y=meta.target_binary.to_numpy()[ids[mask]]
    return y,y,['Benign','DDoS attacks-LOIC-HTTP'],ids[mask],'feb20_matched'

def predict(seed,group,name,part,p):
    import torch,joblib
    from .worker import bind
    start=time.perf_counter()
    if group=='supervised':
        from . import vendor_supervised as m
        bind(m,p);m.model_path=lambda n:p/'models'/(n+('.ubj' if n=='xgboost' else '.joblib'))
        model=m.load_model(name);loaded=time.perf_counter()-start
        score,timing=m.predict_partition(model,name,part)
    elif group=='temporal':
        from . import vendor_temporal as m
        bind(m,p);m.PROTOCOL=SPEC['protocols']['temporal']
        config=read(p/'configs'/f'{name}.json');model=m.load(config);loaded=time.perf_counter()-start
        begin=time.perf_counter();x=m.features(part);preparation=time.perf_counter()-begin
        ids=m.cohort(part)['partition_index'];score,timing=m.predict(model,x,ids,config['length'])
        timing['feature_preparation_seconds']=preparation;timing['complete_inference_seconds']=preparation+timing['inference_seconds']
    elif group=='graph':
        from . import vendor_graph as m
        bind(m,p);config=read(p/'metrics'/f'{name}_training.json')['config']
        model=m.make_model(config);model.load_state_dict(torch.load(p/'models'/f'{name}.pt',weights_only=True));loaded=time.perf_counter()-start
        begin=time.perf_counter();data=m.GraphData();prep=time.perf_counter()-begin
        score,timing=m.predict(model,data,part,1);timing['metadata_loading_seconds']=prep
    elif group=='graph_temporal':
        from . import vendor_graph_temporal as m
        bind(m,p);m.PROTOCOL=SPEC['protocols']['graph_temporal']
        model=m.TemporalHead(336);model.load_state_dict(torch.load(p/'models'/f'{name}.pt',weights_only=True));loaded=time.perf_counter()-start
        representation=np.load(p/'metrics/representations.npy',mmap_mode='r');history=np.load(p/'metrics/history_indices.npy')
        positions=np.flatnonzero(np.load(p/'metrics/target_partitions.npy')==('train','validation','test').index(part))
        score,elapsed=m.predict(model,representation,history,positions,int(name.rsplit('L',1)[1]))
        shared=next(r for r in read(p/'metrics/representation_manifest.json')['timings'] if r['partition']==part)
        history_cost=next(r['history_index_seconds'] for r in read(OUT/'audit/history_replay.json')['timings'] if r['partition']==part)
        if name.endswith('L1'):history_cost=0.
        timing=dict(head_lookup_inference_seconds=elapsed,shared_graph_representation_seconds=shared['representation_seconds'],shared_history_index_seconds=history_cost,
                    inference_seconds=elapsed+shared['representation_seconds'],preparation_inclusive_seconds=elapsed+shared['representation_seconds']+history_cost)
    else:
        from . import vendor_anomaly as m
        bind(m,p);m.PROTOCOL=SPEC['protocols']['anomaly'];model=m.Autoencoder([128,32,128]);model.load_state_dict(torch.load(p/'models'/f'{name}.pt',weights_only=True));loaded=time.perf_counter()-start
        from .vendor_anomaly_data import partition_batches
        begin=time.perf_counter();chunks=[];compute=0.
        for item in partition_batches(part):
            x=item[0] if isinstance(item,tuple) else item
            t=time.perf_counter();chunks.append(m.errors(model,x));compute+=time.perf_counter()-t
        score=np.concatenate(chunks);timing=dict(inference_seconds=time.perf_counter()-begin,prediction_only_seconds=compute)
        if part=='validation':
            from .vendor_anomaly_data import validation_benign
            exact=m.errors(model,validation_benign());original=np.load(p/'metrics'/f'{name}_benign_validation.npy')
            require(np.array_equal(exact,original),'AE benign checkpoint replay mismatch')
            y=cohort(group,part)[0];drift=float(np.max(np.abs(score[y==0]-exact)/(1+np.abs(exact))))
            require(drift<1e-6,'AE mixed-layout reconstruction drift')
            score[y==0]=exact;timing['benign_replay_relative_drift']=drift
    timing['model_loading_seconds']=loaded
    require(np.isfinite(score).all(),'Nonfinite final scores')
    return score,timing

def rows_for(seed,group,name,part,score,locked):
    y,f,book,ids,cohort_name=cohort(group,part);require(len(y)==len(score),'Score/cohort length mismatch')
    ranks=ranking(y,score);metrics=[];families=[]
    for r in locked:
        base=dict(seed=seed,model=name,cohort=cohort_name,partition=part,criterion=r['criterion'],threshold=r['threshold'])
        metrics.append(dict(**base,**counts(y,score,r['threshold']),**ranks,ranking_basis='continuous_score'))
        for i,family in enumerate(book):
            if family=='Benign':continue
            mask=f==i;n=int(mask.sum());detected=int((mask&(score>=r['threshold'])).sum())
            families.append(dict(**base,attack_family=family,N=n,detected=detected,recall=detected/n if n else None))
    return metrics,families,ids

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);parser.add_argument('--group',required=True);parser.add_argument('--model',required=True);a=parser.parse_args()
    check_lock();p=folder(a.seed,a.group);require(not (p/'metrics'/f'{a.model}_evaluation.json').exists(),'Evaluation already exists')
    from src.phase4.train import MemoryMonitor
    from threadpoolctl import threadpool_limits
    train=read(p/'metrics'/f'{a.model}_training.json');checkpoint=p/'models'/(a.model+('.ubj' if a.model=='xgboost' else '.joblib' if a.group=='supervised' else '.pt'))
    lockpath=p/'configs'/f'{a.model}_threshold_lock.json';allmetrics=[];allfamilies=[];artifacts=[];timings=[]
    if a.group=='anomaly':
        benign=np.load(p/'metrics'/f'{a.model}_benign_validation.npy')
        locked=thresholds(np.zeros(len(benign),dtype=np.uint8),benign,'anomaly')
        write(p/'configs'/f'{a.model}_benign_calibration_lock.json',dict(seed=a.seed,thresholds=locked,checkpoint_sha256=sha(checkpoint),benign_score_sha256=sha(p/'metrics'/f'{a.model}_benign_validation.npy'),frozen_before_malicious_scoring=True))
    with threadpool_limits(limits=2),MemoryMonitor() as memory:
        for part in ('validation','test'):
            if part=='test':
                saved=read(lockpath);require(saved['checkpoint_sha256']==sha(checkpoint),'Checkpoint changed after calibration')
                require(saved['validation_scores_sha256']==sha(p/'metrics'/f'{a.model}_validation_scores.npy'),'Validation scores changed after calibration')
            score,timing=predict(a.seed,a.group,a.model,part,p)
            scorepath=p/'metrics'/f'{a.model}_{part}_scores.npy';np.save(scorepath,score)
            if part=='validation':
                y=cohort(a.group,part)[0]
                if a.group!='anomaly':locked=thresholds(y,score,a.group)
                write(lockpath,dict(seed=a.seed,model=a.model,checkpoint_sha256=sha(checkpoint),validation_scores_sha256=sha(scorepath),cohort_ids_sha256=__import__('hashlib').sha256(np.asarray(cohort(a.group,part)[3],dtype='<i8').tobytes()).hexdigest(),thresholds=locked,
                    frozen_before_test=True,spec_sha256=sha(ROOT/'results/final_spec_v1/final_experiment_spec.json')))
                if a.group=='supervised':
                    my,mf,mb,mids,mcn=cohort('temporal','validation')
                    write(p/'configs'/f'{a.model}_matched_threshold_lock.json',dict(seed=a.seed,thresholds=thresholds(my,score[mids],'supervised'),validation_scores_sha256=sha(scorepath),matched_validation_cohort_sha256=sha(ROOT/'results/temporal_v1/metrics/validation_cohort.parquet'),frozen_before_test=True))
            metrics,families,ids=rows_for(a.seed,a.group,a.model,part,score,locked)
            idpath=p/'metrics'/f'{a.model}_{part}_ids.npy';np.save(idpath,ids)
            allmetrics+=metrics;allfamilies+=families;timings.append(dict(partition=part,**timing))
            artifacts.extend(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)) for path in (scorepath,idpath))
    csvwrite(p/'metrics'/f'{a.model}_metrics.csv',allmetrics);csvwrite(p/'metrics'/f'{a.model}_families.csv',allfamilies)
    write(p/'metrics'/f'{a.model}_evaluation.json',dict(seed=a.seed,model=a.model,status='complete',artifacts=artifacts,timings=timings,peak_memory_bytes=memory.peak,checkpoint_sha256=sha(checkpoint),threshold_lock_sha256=sha(lockpath)))
    check_lock();mark('evaluation_complete',a.seed,a.model)

if __name__=='__main__':main()
