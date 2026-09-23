"""Preparation only: saved-score joins, deterministic selection, protocol and audit."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd

SEED=42
PROTECTED=('results/final_spec_v1','results/final_runs_v1','results/baseline_v1','results/graph_v1','data/graph_v1')
CRITERIA={'rf':'fpr_at_most_0.01','ae':'benign_fpr_at_most_0.01','gat':'fpr_at_most_0.01','self_only':'fpr_at_most_0.01'}
MODELS={'rf':('supervised','random_forest','joblib'),'ae':('anomaly','autoencoder_B','pt'),
        'gat':('graph','gatv2','pt'),'self_only':('graph','gatv2_self_only','pt')}
OUTPUTS=('protocol/XAI_PROTOCOL.md','protocol/xai_protocol.json','protocol/case_selection_protocol.json',
 'protocol/feature_group_mapping.csv','protocol/xai_dependency_plan.md','manifests/rf_ae_seed42_decisions.parquet',
 'manifests/gat_seed42_decisions.parquet','manifests/selected_cases.csv','audit/artifact_hash_manifest.csv',
 'audit/integrity_report.md','audit/prediction_replay_plan.md','PHASE_12A_PROTOCOL_SUMMARY.md')
REPLAY={'method':'GraphData.snapshot(group, 1)','window_minutes':1,'cutoff_second':30,
 'completion_rule':'completion_us < cutoff * 1000000','node_aggregates':'all eligible within-window history before message capping',
 'message_cap':8192,'message_selection':'most recent completion; source_row ascending tie order, retain last 8192',
 'uncapped_order':'original chronological metadata order','directed_edges':True,'parallel_edges':True,
 'self_loops':'explicit model self-loops with zero edge attributes; retained during message-edge masking',
 'partition_reset':True,'preserve_node_pruning_and_target_group':True,
 'probability_atol':1e-7,'probability_rtol':0.,'on_failure':'STOP for that case; no explanation',
 'execution_in_phase12a':False,'aggregate_reconstruction_from_capped_edges':False}
SELECTION={'seed':42,'partition':'test','quantiles':[.1,.5,.9],'sort':'ascending score, target_id, source_row',
 'rank':'q*(N-1); nearest unused integer rank, lower rank on equal distance',
 'small_categories':'stop after min(3,N) distinct IDs; quantiles processed .1,.5,.9',
 'empty':'one EMPTY status row, null IDs; no replacement','complementarity_score':{'RF_ONLY_ATTACK':'rf_score',
 'AE_ONLY_ATTACK':'ae_score','RF_AND_AE_ATTACK':'rf_score'},'explanation_inputs':False}

def require(ok,message):
    if not ok:raise ValueError(message)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def snapshot(root):
    result={}
    for folder in PROTECTED:
        p=root/folder;require(p.is_dir(),'Missing protected directory: '+folder)
        for f in sorted(p.rglob('*')):
            if f.is_file():result[str(f.relative_to(root))]=digest(f)
    return result

def verify_snapshot(root,baseline):
    require(snapshot(root)==baseline,'Protected file set or bytes changed')

def dependencies(version=importlib.metadata.version):
    result={}
    for name in ('shap','captum','torch-geometric'):
        try:result[name]=version(name)
        except importlib.metadata.PackageNotFoundError:result[name]='UNAVAILABLE'
    return result

def threshold(lock,role):
    require(lock['seed']==42 and lock['model']==MODELS[role][1] and lock['frozen_before_test'],'Wrong threshold provenance')
    rows=[x for x in lock['thresholds'] if x['criterion']==CRITERIA[role]]
    require(len(rows)==1 and np.isfinite(rows[0]['threshold']),'Missing/invalid frozen criterion')
    return rows[0]['threshold']

def join_scores(metadata,ids_a,scores_a,ids_b,scores_b,role_a,role_b,threshold_a,threshold_b,graph=False):
    a=np.asarray(ids_a);b=np.asarray(ids_b)
    require(a.ndim==b.ndim==1 and a.dtype.kind in 'iu' and b.dtype.kind in 'iu','IDs must be integer vectors')
    require(np.array_equal(a,b) and len(np.unique(a))==len(a),'ID equality/uniqueness failure')
    require(len(a)==len(scores_a)==len(scores_b) and len(a)>0,'Score length mismatch')
    require(np.isfinite(scores_a).all() and np.isfinite(scores_b).all(),'Nonfinite scores')
    require((a>=0).all() and (a<len(metadata)).all(),'ID out of range')
    if graph:
        require(np.array_equal(a,np.flatnonzero(metadata['is_target_test'].to_numpy())),'Graph test cohort/order mismatch')
    else:require(np.array_equal(a,np.arange(len(metadata))),'Full dataset membership/order mismatch')
    df=metadata.iloc[a].copy().reset_index(drop=True)
    require(df.true_label.isin([0,1]).all(),'Invalid labels')
    df.insert(0,'target_id',a)
    if graph:
        df.insert(1,'metadata_index',a);df=df.drop(columns=['is_target_test'])
    for role,score,t in [(role_a,scores_a,threshold_a),(role_b,scores_b,threshold_b)]:
        df[role+'_score']=score;df[role+'_threshold']=t;df[role+'_decision']=(np.asarray(score)>=t).astype('uint8')
        df[role+'_criterion']=CRITERIA[role]
    return df

def categories(rf,gat):
    y=rf.true_label.eq(1);r=rf.rf_decision.eq(1);a=rf.ae_decision.eq(1)
    result={}
    for family,suffix in [('Bot','BOT'),('Infilteration','INFILTERATION')]:
        for label,decision in [('TP',r),('FN',~r)]:result[f'RF_{label}_{suffix}']=(rf[y & rf.attack_family.eq(family)&decision],'rf_score')
    result.update(RF_FP_BENIGN=(rf[~y&r],'rf_score'),RF_ONLY_ATTACK=(rf[y&r&~a],'rf_score'),
        AE_ONLY_ATTACK=(rf[y&~r&a],'ae_score'),RF_AND_AE_ATTACK=(rf[y&r&a],'rf_score'))
    y=gat.true_label.eq(1);g=gat.gat_decision.eq(1);s=gat.self_only_decision.eq(1)
    result.update(GAT_TP=(gat[y&g],'gat_score'),GAT_FN=(gat[y&~g],'gat_score'),GAT_FP=(gat[~y&g],'gat_score'),
        GAT_POS_SELF_NEG=(gat[g&~s],'gat_score'),GAT_NEG_SELF_POS=(gat[~g&s],'gat_score'))
    return result

def select_cases(frame,score,category):
    # Ignore all non-protocol columns, including any hypothetical explanation data.
    d=frame[['target_id','source_row',score]].sort_values([score,'target_id','source_row'],kind='mergesort').reset_index(drop=True)
    require(d.target_id.is_unique,'Duplicate candidate IDs')
    if d.empty:return [dict(category=category,status='EMPTY',available=0,quantile=None,target_id=None,source_row=None,score=None)]
    used=set();rows=[]
    for q in SELECTION['quantiles']:
        if len(used)==len(d):break
        rank=min((i for i in range(len(d)) if i not in used),key=lambda i:(abs(i-q*(len(d)-1)),i))
        used.add(rank);r=d.iloc[rank]
        rows.append(dict(category=category,status='SELECTED',available=len(d),quantile=q,target_id=int(r.target_id),
            source_row=int(r.source_row),score=float(r[score])))
    return rows

def feature_groups(names):
    require(len(names)==80 and len(set(names))==80,'Expected exact 80 unique features')
    groups={
      'protocol_port': [1,78,79,80], 'duration_interarrival':[2,*range(17,31)],
      'volume_packet_counts':[3,4,5,6,52,56,57,59,60,62,63,64,65,68],
      'packet_lengths':[*range(7,15),39,40,41,42,43,53,54,55,69],
      'rates':[15,16,37,38,58,61], 'tcp_flags':[31,32,33,34,*range(44,52)],
      'header_window':[35,36,66,67], 'active_idle':list(range(70,78))}
    require(sorted(i for v in groups.values() for i in v)==list(range(1,81)),'Feature groups must partition inputs')
    return [dict(index=i,feature=names[i-1],group=g,perturbation_unit='protocol_one_hot' if i>=78 else names[i-1])
            for g,indices in groups.items() for i in indices]

def protocol():
    return dict(phase='12A',preparation_only=True,primary_seed=42,case_selection=SELECTION,graph_replay=REPLAY,
        criteria=CRITERIA,dependencies=dependencies(),rf_target='malicious-class probability',
        rf_method='proposed SHAP 0.48.0 TreeExplainer interventional probability; no explainer called in 12A',
        background='Deferred materialization: sorted TRAIN partition IDs, default_rng(42), draw min(500,N) benign then malicious without replacement; no refill; sort final IDs',
        global_cohort='Deferred materialization: sorted test IDs; default_rng(42); min(1000,N) benign then malicious. Malicious family quotas use proportional largest remainder; family-name ties ascending; sample within sorted families; no refill',
        feature_group_aggregation='sum per-feature mean absolute SHAP; distinguish from mean absolute signed group sum',
        references='GAT numeric current-flow zero in frozen standardized space; Protocol grouped to frozen TRAIN mode (6); node log1p aggregates zero (feature perturbation, not realizable graph removal)',
        gat_target='report logit and probability deltas; rank by signed probability drop descending, deterministic component-ID ties',
        message_masking='remove message entries only; retain full-history node aggregates, current target and explicit self loops; never claim historical-flow counterfactual',
        fidelity_k=[1,3,5,10],stability_k=[5,10],stability_seeds=[123,456,789,1024],
        fidelity='top and bottom k signed-ranked components, clip k to availability and deduplicate; report score drop, frozen-threshold flip and k/component_count',
        stability='same seed42 IDs, Spearman with average tie ranks (null for constant ranks), top-k Jaccard; no reselection',
        shap_additivity=dict(atol=1e-6,rtol=1e-5,on_failure='STOP case'),
        forbidden=['training','model loading/inference','SHAP generation','graph perturbation','edge masking','package installation'],
        interpretation='model attribution and sensitivity only; no causal claims; no appearance-based tuning')

def read_json(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def saved(root,role,spec_hash):
    group,model,ext=MODELS[role];p=root/f'results/final_runs_v1/seed_42/{group}'
    lock=read_json(p/f'configs/{model}_threshold_lock.json');ev=read_json(p/f'metrics/{model}_evaluation.json')
    t=threshold(lock,role)
    require(ev['status']=='complete' and ev['seed']==42 and ev['model']==model,'Evaluation provenance mismatch')
    require(lock['spec_sha256']==spec_hash,'Spec mismatch')
    require(digest(p/f'models/{model}.{ext}')==lock['checkpoint_sha256']==ev['checkpoint_sha256'],'Checkpoint hash mismatch')
    require(digest(p/f'configs/{model}_threshold_lock.json')==ev['threshold_lock_sha256'],'Threshold hash mismatch')
    artifacts={str(Path(r['path'])):r['sha256'] for r in ev['artifacts']}
    arrays=[]
    for kind in ('ids','scores'):
        path=p/f'metrics/{model}_test_{kind}.npy'
        require(digest(path)==artifacts[str(path.relative_to(root))],'Score/ID hash mismatch')
        arrays.append(np.load(path,allow_pickle=False))
    return *arrays,t

def run(root):
    root=Path(root).resolve();out=root/'results/xai_v1'
    require(not out.exists(),'Output already exists; refuse overwrite')
    before=snapshot(root)
    specpath=root/'results/final_spec_v1/final_experiment_spec.json';spec=read_json(specpath)
    integrity=read_json(root/'results/final_spec_v1/freeze_integrity.json')
    for name,h in integrity['final_output_hashes'].items():require(digest(root/'results/final_spec_v1'/name)==h,'Frozen specification mismatch')
    names=spec['data']['full_preprocessing']['output_feature_names']
    require(names==read_json(root/'results/baseline_v1/preprocessing.json')['output_feature_names'],'Feature order mismatch')
    mpath=root/'data/baseline_v1/membership/test.parquet'
    membership_hash=digest(mpath)
    require(membership_hash==spec['data']['full_split']['partitions']['test']['membership_sha256'],'Membership hash mismatch')
    membership=pd.read_parquet(mpath).rename(columns={'attack_label':'attack_family','target_binary':'true_label'})
    membership=membership[['source_file','source_row','timestamp','attack_family','true_label']]
    require(((membership.attack_family=='Benign')==(membership.true_label==0)).all(),'Family/label disagreement')
    rf=saved(root,'rf',digest(specpath));ae=saved(root,'ae',digest(specpath))
    rdf=join_scores(membership,rf[0],rf[1],ae[0],ae[1],'rf','ae',rf[2],ae[2])
    gm=read_json(root/'results/graph_v1/metrics/data_manifest.json')
    for item in gm['artifacts']:require(digest(Path(item['path']))==item['sha256'],'Graph artifact mismatch')
    meta=pd.read_parquet(root/'data/graph_v1/metadata.parquet')
    target_ids=np.load(root/'data/graph_v1/target_ids.npy',allow_pickle=False)
    expected=target_ids[meta.partition.to_numpy()[target_ids]==2]
    gat=saved(root,'gat',digest(specpath));self_only=saved(root,'self_only',digest(specpath))
    require(np.array_equal(gat[0],expected),'Frozen sampled target order mismatch')
    meta['timestamp']=pd.to_datetime(meta.second,unit='s');meta['is_target_test']=False
    meta.loc[expected,'is_target_test']=True
    meta=meta.rename(columns={'target_binary':'true_label'})[['source_row','timestamp','true_label','is_target_test']]
    gdf=join_scores(meta,gat[0],gat[1],self_only[0],self_only[1],'gat','self_only',gat[2],self_only[2],graph=True)
    selected=[row for name,(frame,score) in categories(rdf,gdf).items() for row in select_cases(frame,score,name)]
    plan=protocol();mapping=feature_groups(names)
    verify_snapshot(root,before)
    for directory in ('protocol','manifests','audit'):(out/directory).mkdir(parents=True,exist_ok=False)
    def js(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    js('protocol/xai_protocol.json',plan);js('protocol/case_selection_protocol.json',SELECTION)
    pd.DataFrame(mapping).sort_values('index').to_csv(out/'protocol/feature_group_mapping.csv',index=False)
    rdf.to_parquet(out/'manifests/rf_ae_seed42_decisions.parquet',index=False)
    gdf.to_parquet(out/'manifests/gat_seed42_decisions.parquet',index=False)
    pd.DataFrame(selected).to_csv(out/'manifests/selected_cases.csv',index=False)
    (out/'protocol/XAI_PROTOCOL.md').write_text('# Phase12A frozen preparation protocol\n\nThe authoritative machine-readable details are xai_protocol.json, case_selection_protocol.json and feature_group_mapping.csv. Primary seed 42; saved test predictions only. No explanations or model replay occur in 12A. Background/global-cohort sampling algorithms are frozen, with array materialization deferred to a separately authorized step because the Phase12A output list excludes them.\n\n'+json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    (out/'protocol/xai_dependency_plan.md').write_text('# Dependency proposal — approval required\n\nProposed SHAP: 0.48.0. PyPI release metadata lists Python 3.13 support; scikit-learn tree ensembles are supported. Sources: https://pypi.org/pypi/shap/0.48.0/json and https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html . This is metadata-level feasibility, not demonstrated binary or checkpoint compatibility.\n\nProposed separate environment: .venv-xai, Python 3.13.5, numpy 2.1.3, scipy 1.15.3, scikit-learn 1.7.2, pandas 2.2.3, joblib 1.4.2; SHAP 0.48.0. All transitive dependencies and platform wheels must be resolved and locked after approval without changing any modeling dependency. Do not install into base Anaconda or modeling overlays. No installation is performed by this runner. Captum/PyG are unnecessary for the proposed native masking interface and are not requested.\n\nDetected metadata: '+json.dumps(plan['dependencies'])+'\n\nSTOP before installation or any explanation; separate dependency approval and compatibility smoke testing required.\n',encoding='utf-8')
    (out/'audit/prediction_replay_plan.md').write_text('# Future replay plan — not executed\n\nRF: load unchanged same-seed checkpoint only in a later authorized step; exact 80-feature transform and malicious class probability; compare saved score at atol=1e-7, rtol=0. GAT: '+json.dumps(REPLAY,indent=2)+'\n\nMap target metadata index into target_ids position and entire target-minute group. Use the preserved node pruning/order, directed parallel edges, full-history aggregates and model self-loops. eval mode, CPU, two threads; no parameter updates. Readable IPs are unnecessary. Stop a case on mismatch before explanation.\n',encoding='utf-8')
    verify_snapshot(root,before)
    require(digest(mpath)==membership_hash,'Membership changed during preparation')
    rows=[dict(path=p,sha256=h,role='protected_input') for p,h in sorted(before.items())]
    rows.append(dict(path=str(mpath.relative_to(root)),sha256=membership_hash,role='membership_input'))
    rows += [dict(path=str(p.relative_to(root)),sha256=digest(p),role='runner_source') for p in sorted((root/'src/xai_prep').glob('*.py'))]
    (out/'audit/integrity_report.md').write_text('# Integrity PASS\n\nProtected directory file sets and SHA256 hashes unchanged before/after preparation. Frozen spec, membership, graph inputs, checkpoint, score/ID and threshold hashes checked. No model/explanation execution. Artifact manifest excludes its own hash.\n',encoding='utf-8')
    (out/'PHASE_12A_PROTOCOL_SUMMARY.md').write_text('# Phase12A complete\n\nProtocol and saved-prediction manifests frozen; deterministic cases selected without explanation inputs. No model training, inference, explanations, perturbations, or installations. All required outputs are present. Dependency approval is required before later XAI.\n',encoding='utf-8')
    rows += [dict(path=str(p.relative_to(root)),sha256=digest(p),role='phase12a_output') for p in sorted(out.rglob('*')) if p.is_file()]
    pd.DataFrame(rows).to_csv(out/'audit/artifact_hash_manifest.csv',index=False)
    require({str(p.relative_to(out)).replace('\\','/') for p in out.rglob('*') if p.is_file()}==set(OUTPUTS),'Unexpected output set')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true',required=True)
    args=parser.parse_args()
    run(Path(__file__).resolve().parents[2])

if __name__=='__main__':main()
