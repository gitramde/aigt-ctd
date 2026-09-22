"""Freeze final specifications using only file reads, hashing and document writes.

No training modules are imported and no model/optimizer is constructed.
"""
import csv,hashlib,json,os,platform,time
from copy import deepcopy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/final_spec_v1'
SEEDS=[42,123,456,789,1024]
PHASES={'baseline_v1':'baseline','temporal_v1':'temporal','graph_v1':'graph','anomaly_v1':'anomaly',
        'fusion_v1':'fusion','graph_temporal_v1':'graph_temporal'}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,value):Path(p).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def write_csv(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def relative(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def artifact(rel):
    p=ROOT/rel
    return dict(path=rel,sha256=sha(p),bytes=p.stat().st_size)

def verify_development():
    expected={};references={};inventory=set()
    def add(path,digest,origin):
        p=Path(path).resolve()
        require(p.is_file(),'Referenced artifact missing: '+str(p))
        require(p not in expected or expected[p]==digest,'Conflicting historic digests: '+str(p))
        expected[p]=digest;references.setdefault(p,set()).add(origin)
    def walk(obj,origin):
        if isinstance(obj,list):
            for value in obj:walk(value,origin)
        elif isinstance(obj,dict):
            if isinstance(obj.get('path'),str) and isinstance(obj.get('sha256'),str):add(obj['path'],obj['sha256'],origin)
            for key,value in obj.items():
                if key.endswith('_path') and isinstance(value,str) and key[:-5]+'_sha256' in obj:
                    add(value,obj[key[:-5]+'_sha256'],origin)
                if key=='historical_references':
                    for p,digest in value.items():add(p,digest,origin)
                walk(value,origin)
    for folder,source in PHASES.items():
        for p in sorted((ROOT/'results'/folder).rglob('*')):
            if not p.is_file():continue
            inventory.add(p.resolve())
            if p.suffix=='.json':
                obj=read(p);origin=relative(p);walk(obj,origin)
                if isinstance(obj,dict):
                    for key in ('source_code_sha256','source_sha256','implementation_sha256'):
                        if isinstance(obj.get(key),dict):
                            for name,digest in obj[key].items():
                                owner='phase4a' if folder=='baseline_v1' and 'diagnostics' in p.parts else source
                                add(ROOT/'src'/owner/name,digest,origin)
                    if p.name.endswith('_training.json'):
                        stem=p.name.removesuffix('_training.json')
                        if 'model_sha256' in obj:
                            candidates=list((ROOT/'results'/folder/'models').glob(stem+'.*'))
                            if candidates:add(candidates[0],obj['model_sha256'],origin)
                        if 'validation_prediction_sha256' in obj:
                            add(p.parent/(stem+'_validation.npy'),obj['validation_prediction_sha256'],origin)
                        if 'protocol_sha256' in obj:add(ROOT/'results'/folder/'configs/protocol.json',obj['protocol_sha256'],origin)
    for folder in ('src/baseline','src/phase4','src/phase4a','src/temporal','src/graph','src/anomaly','src/fusion','src/graph_temporal',
                   'configs','docs','data/phase4_cache','data/graph_v1','data/anomaly_v1','data/baseline_v1'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:inventory.add(p.resolve())
    for p in ROOT.glob('requirements*.txt'):inventory.add(p.resolve())
    inventory|=set(expected)
    print(f'Hashing {len(inventory)} protected files; {len(expected)} have historical digests',flush=True)
    rows=[]
    for i,p in enumerate(sorted(inventory)):
        digest=sha(p)
        require(p not in expected or digest==expected[p],'Historical hash mismatch: '+str(p))
        rows.append(dict(path=relative(p),sha256=digest,bytes=p.stat().st_size,
            verification='historical_match' if p in expected else 'phase10_snapshot',
            historical_sources=';'.join(sorted(references.get(p,[])))))
        if i%100==0:print('Protected hashes verified:',i+1,flush=True)
    write_csv('artifact_hash_manifest.csv',rows)
    return rows

QUESTIONS=[
    ('RQ1','How effectively do conventional and neural flow-based models detect malicious traffic under chronological distribution shift?'),
    ('RQ2','How effectively do models detect attack families absent from the training period, particularly Bot and Infilteration?'),
    ('RQ3','Does graph-based relational modeling improve DDoS detection over matched non-graph controls on the Feb-20 network-flow graph subset?'),
    ('RQ4','Does explicit temporal context provide incremental benefit over capacity-matched non-temporal controls?'),
    ('RQ5','Does a benign-trained reconstruction anomaly branch provide complementary detection behavior for attack families absent from training?'),
    ('RQ6','What computational overhead is introduced by graph, temporal and anomaly components?')]
CLAIMS=[
    'Conventional supervised models can fit known training distributions extremely well but generalization varies substantially across later attack families.',
    'Graph modeling achieves highly accurate LOIC-HTTP detection on the Feb-20 benchmark, but incremental graph benefit over strong controls is operating-point dependent.',
    'Temporal context did not show a consistent incremental benefit in either the full-dataset temporal experiment or graph-temporal experiment.',
    'Benign-trained reconstruction anomaly detection provides complementary family-specific detections but weak aggregate performance at strict false-positive control.',
    'Simple risk fusion does not establish overall superiority under comparable false-alarm control.']

METRICS='''# Frozen metric definitions

Positive class is malicious; negative class is Benign. Detection is score >= threshold. On each identical cohort and seed, save TN, FP, FN, TP and N as integer counts.

| Metric | Definition |
| --- | --- |
| Accuracy | (TP + TN) / N |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2 TP / (2 TP + FP + FN) |
| Macro-F1 | mean of malicious F1 and 2 TN / (2 TN + FP + FN) |
| FPR | FP / (FP + TN) |
| FNR (supplementary) | FN / (FN + TP) |
| ROC-AUC | sklearn roc_auc_score on the continuous malicious-oriented score, ties handled by its ranking implementation |
| PR-AUC | sklearn auc(recall, precision) from precision_recall_curve; trapezoidal area |
| Average Precision | sklearn average_precision_score; not interchangeable with trapezoidal PR-AUC |
| Family recall | detected malicious rows of the family / all evaluated rows of that family |

For zero denominators, binary classification ratios are 0, matching the frozen metric implementation. Absent family recall is null with N=0, not 0. ROC/PR/AP require both classes; record null and reason otherwise. Raw AE error is a score, not a probability. No clipping or percentile transform is introduced for final AE/OR reporting.

Report each metric for each seed, model, cohort, partition and operating criterion. Report arithmetic mean and sample standard deviation (ddof=1) over the five completed seeds; retain full-precision values, use six decimals for display and scientific notation for small differences. Report number of completed seeds. Do not silently drop failures or substitute development results. A planned five-seed result is incomplete until all five exist. IF remains a single development-seed row without a fabricated standard deviation.

The fixed split and target records are shared across seeds. Seed variation measures training stochasticity only, not dataset or split uncertainty. Five seeds are not five independent dataset replications. Do not pool repeated targets or average confusion counts as independent datasets; retain per-seed confusion matrices. Compute metrics per seed before mean/std aggregation. No significance tests, confidence intervals, or confirmatory claims are predeclared.

For component contribution, save within-seed signed differences and their mean/sample SD: graph minus edge MLP, graph minus self-only if available, temporal-L64 minus temporal-L1, and graph-temporal-L8 minus graph-temporal-L1. Include recall, precision, F1, macro-F1, FPR, ROC-AUC and AP at every shared operating point. Positive FPR differences mean more false alarms. Do not claim incremental benefit from one favorable operating point alone.

Bot and Infilteration diagnostics apply to full-data and Phase5 matched cohorts. Both families are absent from full TRAIN; Infilteration is present in validation. Supervised validation selection is therefore label-aware for this family. Feb20 tests only Benign versus DDoS attacks-LOIC-HTTP, not unseen-family detection. Preserve the literal dataset spelling Infilteration.

OR is a binary decision, not a new fitted ranking score. Its ROC-AUC, trapezoidal PR-AUC and AP, if displayed to complete the metric schema, use the 0/1 decision and are explicitly marked binary-decision ranking. Do not compare that AP as equivalent to continuous RF/AE ranking AP. Report actual test FPR, Bot/Infilteration recall, and RF-only/AE-only/both/neither detections and Jaccard by family at the component 1% points. Jaccard is intersection/union, null for empty union. Component caps do not guarantee a 1% OR union FPR.
'''

RUNTIME=dict(device='cpu',threads=2,rf_n_jobs=1,concurrent_training_jobs=1,
    hardware=dict(cpu='Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz',physical_cores=4,logical_cpus=8,ram_bytes=16944599040,platform='Windows-11-10.0.22631-SP0'),
    software=dict(python='3.13.5',numpy='2.1.3',pandas='2.2.3',pyarrow='19.0.0',scikit_learn='1.7.2',xgboost='3.0.5',torch='2.6.0+cpu',scipy='1.15.3',matplotlib='3.10.0',psutil='5.9.0',joblib='1.4.2',threadpoolctl='3.5.0'),
    environment_policy='Verify exact modeling versions before fitting; baseline preparation originally used sklearn1.6.1 but is transform-only and never refitted. The Phase10 sandbox inspection resolved sklearn1.6.1; this does not authorize use for final fitting. Read the isolated recorded runtime and fail if version checks disagree.',
    training='Report optimizer/fit wall seconds separately from validation-selection seconds, input/preparation seconds and total stage wall seconds; include necessary XGBoost quantile matrix construction in fitting cost; no subtracting undocumented costs.',
    inference='Report model-only compute, feature transformation/loading, graph-prefix construction/encoding, sequence/history construction and lookup, and complete available-pipeline total separately. Raw cleaning/join/cache materialization/model loading are separately recorded preparation stages, never silently omitted from a claimed end-to-end value.',
    graph_temporal='Same-seed frozen encoder reused by L1/L8. Report shared embedding preparation and shared history-index preparation once, head-only inference and preparation-inclusive total; include diagnostic-count work in its own explicitly labeled replay cost. Do not add repeated shared costs across models.',
    latency='1000 * 1000 * inference_seconds / number_of_evaluated_target_flows_or_edges, in milliseconds per 1000; name the corresponding stage and cohort.',
    memory='Process high-water RSS/Windows peak working set bytes, including dependencies and resident caches; poll every .05 seconds; use a separate worker per fit/evaluation. For separately executed stages report max(stage peaks), not their sum; disclose that as an amortized pipeline maximum.',
    model_size='Checkpoint bytes; report head-only and required pipeline size, counting shared encoder once. Preserve unused serialized parameters caveat for self-only and unused positional slots.',
    comparison='Keep hardware and measurement definitions fixed across final seeds. Never compare full-cohort flow timing, sampled temporal timing, Feb20 graph timing and cached-pipeline timing without explicit scope qualification. Record any hardware change and do not pool incompatible timing runs.',
    repeat_policy='One measured training/evaluation run per seed under the same cache policy; no picking fastest repeats. Repeated values across partition rows are not independent costs.')

def make_spec():
    baseline=read(ROOT/'results/baseline_v1/split_plan.json');pre=read(ROOT/'results/baseline_v1/preprocessing.json')
    graphpre=read(ROOT/'results/graph_v1/configs/preprocessing.json');phase4=read(ROOT/'configs/phase4_seed42.json')
    tp=read(ROOT/'results/temporal_v1/configs/protocol.json');gp=read(ROOT/'results/graph_v1/configs/protocol.json')
    ap=read(ROOT/'results/anomaly_v1/configs/protocol.json');gt=read(ROOT/'results/graph_temporal_v1/configs/protocol.json')
    for protocol in (tp,gp,ap,gt):protocol.pop('seed',None);protocol.pop('phase',None)
    for k in ('alternate_architecture','architecture_search','lengths'):tp.pop(k,None)
    tp['lengths']=[1,64];tp['model_search']='disabled'
    for k in ('candidates','additional_baselines','sensitivity_selection','sensitivity_windows'):gp.pop(k,None)
    gp['architecture_search']='disabled';gp['final_windows']=[1]
    ap.pop('architectures');ap.pop('protocol_B_objectives');ap.pop('isolation_forest')
    ap.update(architecture=[128,32,128],protocol_B='excluded from final calibration',architecture_search='disabled')
    gt['lengths']=[1,8]
    train=baseline['partitions']['train'];n=train['rows'];neg=train['class_counts']['Benign'];pos=n-neg
    weights={'0':n/(2*neg),'1':n/(2*pos)}
    AdamW=dict(betas=[.9,.999],eps=1e-8,amsgrad=False,maximize=False,foreach=None,capturable=False,differentiable=False,fused=None)
    rows=[];models={}
    def model(key,family,cohort,configuration,budget,selection,dependency='',status='five_seed_required'):
        models[key]=dict(family=family,cohort=cohort,configuration=configuration,budget=budget,selection=selection,dependency=dependency,status=status)
        rows.append(dict(model_id=key,family=family,cohort=cohort,final_status=status,seeds=','.join(map(str,SEEDS)),
            dependency=dependency,configuration_json_pointer='/models/'+key,training_in_phase10=False))
    selection=dict(metric='validation macro-F1 at fixed .5',minimum_improvement=1e-5,patience=2,
        rule='score > best + minimum_improvement; save improving checkpoint; ties retain earlier checkpoint; restore best; stop after 2 nonimproving scheduled checks')
    common=dict(dtype='float32 model inputs; frozen float64 transformation before conversion',outer_batch_size=50000,train_rows=n,
        class_weights=weights,seed='s',epoch_shuffle='NumPy default_rng(s+epoch-1).permutation(all TRAIN indices)',test_access_for_selection=False)
    lr=dict(loss='log_loss',penalty='l2',alpha=.0001,learning_rate='constant',eta0=.001,average=True,shuffle=True,random_state='s',
        class_weight=weights,fit_intercept=True,early_stopping=False,epsilon=.1,l1_ratio=.15,power_t=.5,tol=.001,max_iter=1000,n_iter_no_change=5,
        n_jobs=None,validation_fraction=.1,verbose=0,warm_start=False,
        execution='partial_fit each outer batch with classes=[0,1]; max_iter/tol do not replace the five-pass externally controlled schedule')
    model('logistic_regression','supervised','full_dataset',dict(**common,implementation='sklearn.SGDClassifier incremental logistic regression',estimator=lr),dict(max_epochs=5),selection)
    rf=dict(n_estimators_initial=16,tree_checkpoints=[16,32,64],criterion='gini',max_depth=16,min_samples_split=2,min_samples_leaf=20,
        min_weight_fraction_leaf=0.,max_features='sqrt',max_leaf_nodes=None,min_impurity_decrease=0.,bootstrap=True,
        max_samples=min(n,500000),oob_score=False,n_jobs=1,random_state='s',warm_start=True,class_weight=weights,ccp_alpha=0.,monotonic_cst=None,verbose=0,
        execution='fit same full TRAIN cache at each warm-start checkpoint; all rows eligible, each tree bootstrap draws 500000; validation chooses checkpoint')
    model('random_forest','supervised','full_dataset',dict(**common,implementation='sklearn.RandomForestClassifier',estimator=rf),dict(tree_checkpoints=[16,32,64]),selection)
    xgb=dict(objective='binary:logistic',booster='gbtree',tree_method='hist',device='cpu',seed='s',seed_per_iteration=True,nthread=2,
        scale_pos_weight=neg/pos,max_depth=6,max_bin=128,eta=.1,subsample=.8,colsample_bytree=.8,reg_lambda=1.,reg_alpha=0.,
        min_child_weight=1.,gamma=0.,max_delta_step=0.,grow_policy='depthwise',sampling_method='uniform',
        base_score='XGBoost3.0.5 default automatic intercept estimation; no manual tuning',
        execution='QuantileDMatrix from 50000-row CacheIterator; max_bin128,nthread2; callback checks only at 20/40/60/80/100 rounds')
    model('xgboost','supervised','full_dataset',dict(**common,implementation='xgboost.train',estimator=xgb),dict(round_checkpoints=[20,40,60,80,100]),selection)
    mlp=dict(hidden_layer_sizes=[64,32],activation='relu',solver='adam',alpha=.0001,batch_size=2048,learning_rate='constant',learning_rate_init=.001,
        shuffle=True,random_state='s',early_stopping=False,beta_1=.9,beta_2=.999,epsilon=1e-8,tol=.0001,max_iter=200,
        n_iter_no_change=10,validation_fraction=.1,warm_start=False,momentum=.9,nesterovs_momentum=True,power_t=.5,max_fun=15000,verbose=False,
        sample_weight='TRAIN inverse-frequency weight per row in every partial_fit call',
        execution='partial_fit(classes=[0,1]) on each 50000-row outer batch; internal batch2048; external five-pass schedule and validation checkpointing')
    model('mlp','supervised','full_dataset',dict(**common,implementation='sklearn.MLPClassifier',estimator=mlp),dict(max_epochs=5),selection)
    for length in (1,64):
        config=read(ROOT/f'results/temporal_v1/configs/{"transformer_L1_control" if length==1 else "transformer_L64_d64_n2_p01"}.json')
        config['seed']='s'
        config.update(input_features=len(pre['output_feature_names']),feedforward_dim=128,activation='gelu',norm_first=True,
            final_norm='LayerNorm64 eps1e-5',layer_norm_eps=1e-5,bias=True,batch_first=True,enable_nested_tensor=False,
            position='fixed sinusoid,64 slots; even sin odd cos; exp(arange(0,64,2)*(-log10000/64)); current position63 for both lengths',
            initialization='PyTorch defaults for scalar/vector parameters; Xavier uniform for every parameter with dimension>1; positional encoding is a buffer',
            output='linear64->1 on last token; sigmoid scores',attention='bidirectional within supplied start-ordered historical/current window; no future-index flow',
            optimizer=dict(name='AdamW',lr=.0003,weight_decay=.01,**AdamW),loss='BCEWithLogitsLoss(pos_weight=85086/14876)',gradient_clip_norm=1.,
            batch_size=256,epoch_shuffle='default_rng(s+epoch-1).permutation(number of fixed TRAIN targets)',dtype='float32',deterministic_algorithms=True)
        model(f'transformer_L{length}','temporal','phase5_matched',config,dict(max_epochs=3),selection)
    graphnetwork=dict(node_input_dim=8,history_edge_dim=9,hidden=128,layers=2,heads=4,channels_per_head=32,dropout=.2,
        node_projection='Linear8->128 with bias',node_features=gp['node_features'],
        attention='edge-aware dynamic GATv2; source/destination/edge linear projections without bias; per-head learnable attention Xavier uniform; LeakyReLU slope .2; stable softmax over incoming edges; attention dropout .2',
        loops='explicit self loops with zero edge attributes for all retained nodes',
        layer_update='LayerNorm(x + Dropout(ELU(GATv2(x,edges,history_features)))); LayerNorm eps1e-5, affine; two layers',
        classifier=f'concat(src128,dst128,current{len(graphpre["output_feature_names"])})->Linear128->GELU->Dropout.2->Linear1',
        initialization='PyTorch2.6.0 defaults except Xavier uniform edge-attention vectors and zero GAT output bias',
        optimizer=dict(name='AdamW',lr=.0005,weight_decay=.01,**AdamW),loss='BCEWithLogitsLoss(pos_weight=168357/4627)',
        batch='one entire target-minute graph per update',epoch_shuffle='default_rng(s+epoch-1).permutation(TRAIN graph groups)',
        gradient_clip_norm=1.,dtype='float32',deterministic_algorithms=True)
    edge=dict(input_features=len(graphpre['output_feature_names']),network='Linear(input,298)->GELU->Dropout.2->Linear(298,298)->GELU->Dropout.2->Linear(298,1)',
        optimizer=graphnetwork['optimizer'],loss=graphnetwork['loss'],batch=graphnetwork['batch'],epoch_shuffle=graphnetwork['epoch_shuffle'],
        gradient_clip_norm=1.,dtype='float32',initialization='PyTorch default Linear initialization',deterministic_algorithms=True)
    model('edge_mlp','graph_control','feb20_matched',edge,dict(max_epochs=4),selection)
    model('gatv2','graph','feb20_matched',graphnetwork,dict(max_epochs=4),selection)
    selfonly=deepcopy(graphnetwork);selfonly['inter_node_messages']='empty edges/features; explicit self loops remain; original historical node inputs retained'
    selfonly['capacity_caveat']='unused edge-attention parameters remain serialized; equal stored count is not equal active degrees of freedom'
    model('gatv2_self_only','graph_ablation','feb20_matched',selfonly,dict(max_epochs=4),selection,status='five_seed_planned_resource_conditional')
    for length in (1,8):
        conf=dict(length=length,encoder='same-seed final gatv2 best validation checkpoint; requires_grad=False and eval mode',
            representation=f'concat(src128,dst128,current{len(graphpre["output_feature_names"])}); dimension {256+len(graphpre["output_feature_names"])}',
            projection=f'Linear({256+len(graphpre["output_feature_names"])},64)',d_model=64,layers=2,heads=4,feedforward_dim=128,dropout=.1,
            activation='gelu',norm_first=False,layer_norm_eps=1e-5,bias=True,batch_first=True,final_encoder_norm=None,enable_nested_tensor=False,
            positions='learned (1,8,64) Normal(mean0,std.02); use last L slots; current always slot7; do not shrink table at L1',
            initialization='PyTorch default Linear/Transformer layers; TransformerEncoder clones initial layer unchanged, exactly Phase9; learned positions Normal(0,.02)',
            attention='key padding mask; padded raw representations zero; current unmasked; bidirectional within already eligible context; only final output classified; no persistent transformed hidden state',
            output='Linear64->1 on last token',optimizer=dict(name='AdamW',lr=.0005,weight_decay=.01,**AdamW),
            loss='BCEWithLogitsLoss(pos_weight=168357/4627)',batch_size=2048,gradient_clip_norm=1.,
            epoch_shuffle='default_rng(s+epoch-1).permutation(all fixed TRAIN target positions)',trainable_parameters=89089,dtype='float32',deterministic_algorithms=True)
        model(f'gat_transformer_L{length}','graph_temporal','feb20_matched',conf,dict(max_epochs=4),selection,dependency='gatv2[s]')
    ae=dict(input_features=len(pre['output_feature_names']),hidden_layers=[128,32,128],output_features=len(pre['output_feature_names']),
        network='Linear80->128->ReLU->Linear128->32->ReLU->Linear32->128->ReLU->Linear128->80; linear output; no dropout',
        initialization='PyTorch default Linear initialization',optimizer=dict(name='AdamW',lr=.001,weight_decay=.00001,**AdamW),
        loss='float32 mean squared reconstruction error',gradient_clip_norm=10.,batch_size=8192,read_block_size=65536,
        population='every one of 10885643 benign TRAIN rows per executed epoch; no AE subsampling',
        order='starts arange(0,full_TRAIN_rows,65536); rng=default_rng(s+epoch-1); permute blocks; within each selected block permute benign row IDs with same RNG; slice 8192 minibatches',
        score='convert reconstructed and input vectors individually to float64, subtract, square, mean across features; greater error more anomalous',
        benign_validation='all 1583520 benign validation records; no malicious validation used for checkpoint/calibration',
        replay='replay checkpoint on exact benign cache batch layout and preserve exact benign calibration scores in full validation array; mixed-layout relative drift must remain <1e-6 with denominator1+abs(score)',
        preprocessing='frozen baseline all-class TRAIN preprocessing, not benign-only fitted preprocessing',deterministic_algorithms=True)
    model('autoencoder_B','anomaly','full_dataset',ae,dict(max_epochs=4),dict(metric='minimum benign-validation mean reconstruction MSE',patience=2,
        minimum_relative_improvement=.0001,rule='mean < best*(1-.0001); save improving checkpoint; ties retain earlier; architecture already fixed; no malicious validation/test selection'))
    models['isolation_forest']=dict(status='development_seed42_secondary_only',cohort='full_dataset',
        configuration=read(ROOT/'results/anomaly_v1/configs/protocol.json')['isolation_forest'],training_authorized=False,
        policy='Use only protected Phase7 seed42 predictions/metrics, explicitly development evidence; no five-seed mean/std or silent promotion')
    rows.append(dict(model_id='isolation_forest',family='anomaly_secondary',cohort='full_dataset',final_status='development_seed42_secondary_only',seeds='42',dependency='existing Phase7',configuration_json_pointer='/models/isolation_forest',training_in_phase10=False))
    models['rf_or_ae']=dict(status='derived_no_training',cohort='full_dataset',dependencies=['random_forest[s]','autoencoder_B[s]'],
        decision='(RF_score >= seed-specific full-validation label-aware FPR<=.01 threshold) OR (AE_error >= seed-specific benign-validation ProtocolA .01 threshold)',
        thresholds='freeze each component before test evaluation; do not recalibrate OR union; do not reuse seed42 numeric thresholds across seeds',
        reporting='RF alone, AE alone and OR on same full test IDs; no score normalization, alpha, calibration model or fitted fusion',
        OR_cap='two approximately1% component caps do not imply1% union cap; explicitly compare actual FPR and family tradeoffs')
    rows.append(dict(model_id='rf_or_ae',family='derived_fusion',cohort='full_dataset',final_status='derived_no_training',seeds=','.join(map(str,SEEDS)),dependency='random_forest[s];autoencoder_B[s]',configuration_json_pointer='/models/rf_or_ae',training_in_phase10=False))
    spec=dict(schema_version=1,specification='final_spec_v1',phase=10,status='frozen_awaiting_explicit_execution_approval',
        created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),model_training_performed=False,execution_authorized=False,seeds=SEEDS,
        seeds_policy='Fresh fits for all five seeds including42; development42 remains immutable and is not substituted into final aggregation. Replace only stochastic seed inputs; same seed across paired components. No architecture/hyperparameter/threshold-rule changes after other seeds.',
        seed_mapping=dict(python_random='s',numpy_global='s',torch_manual_seed='s',sklearn_random_state='s',xgboost_seed='s',
            epoch_order='s+epoch-1',fixed_data_preparation_seed=20260920,data_membership='unchanged across all seeds',deterministic_algorithms=True),
        model_registry=rows,models=models,
        data=dict(cleaning_configuration=read(ROOT/'configs/baseline.json'),cleaning_semantics='Exact full-row duplicates removed per file (hash collision resolved by tuple equality), repeated headers removed, source row IDs retained. Parse %d/%m/%Y %H:%M:%S without AM/PM repair. Invalid/date-mismatched/1970 timestamps quarantined rather than repaired. Missing tokens normalized; infinities only in configured rate columns become missing. Numeric values float64; categorical Protocol string; chronological sort timestamp/source_row within file. Reuse exact cleaned files and partition membership; no cleaning/split rerun.',
            target='0 Benign;1 any other attack label; attack labels/source IDs/timestamps metadata only',
            full_split=baseline,full_preprocessing=pre,graph_preprocessing=graphpre,
            preprocessing_rule='No per-seed refit. Numeric TRAIN means impute; frozen standard scaling, Protocol TRAIN-mode imputation and frozen ordered one-hot levels; unseen category all-zero; ordered frozen feature lists in state. Neither labels nor identifiers enter features.',
            full_dataset_counts={p:d['rows'] for p,d in baseline['partitions'].items()},
            full_temporal_cohort=read(ROOT/'results/temporal_v1/metrics/sequence_manifest.json'),
            graph_split_approval=read(ROOT/'results/graph_v1/configs/split_approval.json'),
            graph_membership_definition=read(ROOT/'results/graph_v1/pre_split/proposed_split.json'),
            graph_temporal_cohort=read(ROOT/'results/graph_temporal_v1/matched_cohort_manifest.json'),
            graph_data_manifest=read(ROOT/'results/graph_v1/metrics/data_manifest.json'),
            baseline_training_cache_manifest=read(ROOT/'data/phase4_cache/training_cache.json'),
            anomaly_training_manifest=read(ROOT/'results/anomaly_v1/anomaly_training_manifest.json')),
        protocols=dict(temporal=tp,graph=gp,graph_temporal=gt,anomaly=ap,
            temporal_semantics='Same-day start-timestamp ordered global adjacent-flow sequences; exclude first63 each day; TRAIN targets every128 from day_start+63, validation/test every8; L1 takes only current at position63; L64 takes target-63 through target; no cross-day/partition sequences. Phase5 is not endpoint-consistent graph history and does not establish completion-causal online availability.',
            temporal_matched_controls='Slice each seed full-data conventional predictions to exact Phase5 validation/test partition_index IDs; independently recalibrate FPR<=1% on that matched validation slice; no control refit; do not borrow full-validation numeric threshold for the slice.',
            graph_semantics='Use exact approved start-time membership before10:30 / [10:30,11:00) / >=11:00. Every minute second30 cutoff; targets start in final30sec, stride16 TRAIN/4 validation-test resetting minute. Current flow features available at completion; within-current-minute history starts before cutoff and completes strictly before cutoff. No partition crossing. Aggregate all history; cap messages8192 most recently completed, ties source_row; prune irrelevant nodes after aggregation; directed source->destination; IP IDs topology only.',
            graph_temporal_semantics='History lookup uses only fixed Phase6 sampled targets in previous calendar-minute slots. Latest eligible directed pair first, same source second, same destination third; eligible completion strictly before current second30 cutoff; ascending source_row breaks completion ties by retaining greatest. L8=current+7 previous minutes, L1=current; missing slots zero/key-masked without time compression; per-partition reset. Same-seed graph embeddings frozen in eval mode; own flow features only after its completion. Retain all original targets.',
            graph_temporal_gate='Reuse exact Phase9 cohort/history indices and hashes. The documented diagnostic-stage gate amendment is fixed now; do not rerun or relax it based on final-seed outcomes.',
            coverage_gate_amendment=read(ROOT/'results/graph_temporal_v1/configs/coverage_gate_amendment.json'),
            graph_split_limit='Start-time membership allows6097 TRAIN flows (253 targets) and5397 validation flows (832 targets) to finish at/after next partition start. Retain membership; no cross-partition history; offline benchmark, not boundary-time deployment replay; no silent purge/embargo.'),
        thresholds=dict(comparator='>=',supervised_primary=['fixed_0_5','validation_fpr_at_most_0.01'],
            temporal_primary=['fixed_0_5','validation_fpr_at_most_0.01'],temporal_secondary=['maximum_macro_f1','validation_fpr_at_most_0.005','validation_fpr_at_most_0.001'],
            graph_and_graph_temporal=['fixed_0_5','maximum_macro_f1','validation_fpr_at_most_0.01'],
            label_aware_algorithm='Candidates unique validation scores descending plus nextafter(max_score,+inf), evaluate >= threshold. Macro-F1: maximize then highest threshold. FPR cap: feasible FP/N_benign<=cap; maximize TP/recall then minimize FP then highest threshold. Use only the evaluated cohort validation labels; never test.',
            anomaly_primary=dict(protocol='A_benign_only',cap=.01),anomaly_secondary_caps=[.05,.005,.001],
            anomaly_algorithm='Sort all benign validation errors ascending b (N>0). k=floor(cap*N), threshold=nextafter(b[N-k-1],+inf). Compare >=; verify false positives<=k. Ties conservatively excluded; raw errors are not probabilities.',
            per_seed='Recalibrate numeric thresholds from that seed validation scores under fixed rules. Freeze model hash, score hash, cohort hash and thresholds before test scoring/evaluation. Test labels never select anything.',
            cap_limitation='Empirical validation cap is not a population/test guarantee; OR union not capped.',anomaly_protocol_B='not authorized for final selection'),
        statistics=dict(seeds=SEEDS,mean='arithmetic per-seed metric mean',std='sample standard deviation ddof1',
            metrics=['accuracy','precision','recall','f1','macro_f1','false_positive_rate','roc_auc','pr_auc','average_precision'],
            supplementary=['false_negative_rate','TN','FP','FN','TP','family_N','family_detected','family_recall'],
            uncertainty='training stochasticity on fixed data, not independent dataset/split replications',test_already_observed=True,
            no_confusion_count_pooling=True,no_significance_claim=True,paired_differences=True),
        runtime=RUNTIME,research_questions=dict(QUESTIONS),claim_freeze=CLAIMS,
        planned_fit_count=dict(required=55,resource_conditional_self_only=5,total_if_all_feasible=60,derived_fusion_rows=5),
        exclusions=['Phase5 L16/L32/alternate architecture search','Phase6 5/10-minute primary windows','Phase9 L4 final fitting',
            'AE architecture A search','five-seed Isolation Forest','MAX/weighted fusion final fitting or selection','anomaly ProtocolB tuning',
            'joint GAT/Transformer training','anomaly integration into graph-temporal','SHAP/GNNExplainer','new features','new split','zero-day empirical claims'],
        self_only_resource_rule='Plan all five self-only seeds. If infeasible, document measured resource reason before omission; do not inspect performance to decide. No changed width/budget, no replacement seed; any partial run remains explicitly incomplete secondary evidence.',
        stopping_rule='Phase10 ends with specification files only. Explicit user approval is required before ANY final training, including the fresh seed42 rerun. Existing development runners have hard-coded42 and old output paths; do not invoke them for final runs. A separate seed-parameterized final runner must enforce this spec without modifying frozen development sources.')
    return spec,rows

RUN_PLAN='''# Final run plan — not executed

Status: specification frozen; execution awaits explicit user approval. No final model was trained in Phase10, including seed42. Seeds123/456/789/1024 were not executed.

1. After explicit approval, build a separate final runner/output tree (proposed `results/final_runs_v1/seed_<s>/`), retaining this spec and every development artifact unchanged. Existing development entry points hard-code seed42, write old paths and may reopen searches; do not launch them as final runners. Copy/adapt only the frozen algorithms into new code, parameterize RNG/epoch shuffle/output paths and remove all architecture searches. Verify behavior with semantic tests before fitting. The freeze itself is not authorization to implement or execute final training now.
2. Verify artifact_hash_manifest.csv, the specification hash, exact library versions, CPU/RAM/device/thread settings, frozen feature dimensions, membership order and all cohort/history IDs. Check invariants and masks. No data or preprocessing refit. Refuse silent version drift: preparation sklearn1.6.1 state is reused transform-only; modeling requires recorded sklearn1.7.2. Snapshot final runner source and effective configs before the first fit. Record and stop on a mismatch.
3. For seeds in order42,123,456,789,1024, train only the frozen four full-data supervised models with their original bounded validation-checkpoint procedures. Fresh seed42 replaces nothing in development. Save configs, stage histories, checkpoints, scores, hashes, TRAIN-derived weights, exact target IDs and environment.
4. For each seed, train only Phase5-style Transformer-L64 and L1 on the frozen matched targets. Filter same-seed full-data conventional predictions to those target IDs and recalibrate their matched-validation FPR threshold. Never refit conventional models to the sampled temporal cohort.
5. For each seed, train the Feb20 Edge MLP and selected GAT architecture from scratch; retain self-only GAT across all five seeds if feasible under the fixed resource rule. Use only the one-minute primary graph. The 5/10-minute seed42 studies remain sensitivity results.
6. Freeze that seed's selected GAT checkpoint, extract that seed's graph representations and fit graph-temporal L1 and L8 heads. Both heads share the same frozen encoder and fixed history indices. Rebuild representations per seed; never reuse Phase9 seed42 embeddings for other seeds. No end-to-end fine-tuning; no L4 run. Shared prepared IDs/history may be reused only after checksum verification.
7. For each seed, train only AE-B on every benign TRAIN record under the frozen block/minibatch ordering and benign-validation reconstruction selection. Lock Protocol-A caps without malicious labels. Do not train AE-A or IF. Retain original IF only as single-seed development evidence.
8. Freeze seed-specific model and threshold locks before test evaluation. Threshold rules and cohorts are fixed; numerical thresholds and validation-selected checkpoints may differ across seeds. Derive RF, AE and OR component comparison from matching-seed full-cohort predictions at their approximately1% validation operating points. No alpha/normalization/union-threshold optimization.
9. Store individual seed metric/family/confusion/runtime rows and paired component deltas. Produce mean and sample SD only after all five required runs exist. Validate outputs by recomputing metrics from saved scores, checking cohort equality and all provenance hashes. Do not pool repeated test rows as independent data. No new model decisions from seed outcomes.
10. Stop after the approved frozen final evaluation. Any subsequent design change requires a separately versioned experiment and explicit approval, with old evidence retained. A resource failure is logged, not resolved by performance-driven seed/model substitution.

Expected workload after approval:55 mandatory fits (11 configurations ×5 seeds), plus5 planned resource-conditional self-only fits. OR adds no fitting. Encoder fits are reused by both graph-temporal heads, not duplicated. IF and other excluded development ablations add no final fits.

Save per seed: effective configuration, source/environment hashes, checkpoint and score hashes, selected epoch/step, full training/validation histories, threshold lock, target IDs, validation/test metrics, per-family N/detected/recall, confusion counts, component deltas, RF–AE overlap/Jaccard, runtime stage measurements, peak memory and serialized sizes. Keep audit logs separate from metric data. No overwriting or automatic resume from a development artifact; final-stage resume requires matching seed/config/input hashes.
'''

def document(spec,registry):
    rq='# Frozen research questions\n\n'+'\n\n'.join(f'**{key}:** {question}' for key,question in QUESTIONS)+'\n\nUse “attack families absent from training” for empirical full-dataset claims. No zero-day-detection claim. RQ3 concerns only the Feb20 within-day binary graph subset.\n'
    claims='# Claim freeze before additional seeds\n\n'+'\n\n'.join(f'{i+1}. {c}' for i,c in enumerate(CLAIMS))
    claims+='\n\nThese are development interpretations, not outcomes that final data must reproduce. Update numerical estimates and report any contrary evidence honestly; do not tune architectures post hoc to change their direction or suppress inconsistent results. The model roster, selection/calibration algorithms and exclusions remain frozen. All test cohorts were previously observed; final multi-seed evaluation measures stochastic stability on these fixed splits, not a new confirmatory holdout.\n'
    (OUT/'research_questions.md').write_text(rq,encoding='utf-8');(OUT/'claim_freeze.md').write_text(claims,encoding='utf-8')
    (OUT/'metric_definitions.md').write_text(METRICS,encoding='utf-8');(OUT/'final_run_plan.md').write_text(RUN_PLAN,encoding='utf-8')
    summary='# FINAL EXPERIMENT SPECIFICATION — final_spec_v1\n\n'
    summary+='**FROZEN; EXECUTION NOT AUTHORIZED. Phase10 performs no training.** Await explicit approval before the final seed42 rerun or seeds123/456/789/1024. All development artifacts are protected.\n\n'
    summary+='## Frozen roster and dependencies\n\n| model | cohort | final status | dependency |\n| --- | --- | --- | --- |\n'
    for r in registry:summary+=f"| {r['model_id']} | {r['cohort']} | {r['final_status']} | {r.get('dependency','')} |\n"
    summary+='\nEvery stochastic configuration is freshly fitted under42,123,456,789,1024 with the same bounded training/validation rules. Seed42 development is not reused as a final fit. Architecture search is disabled; checkpoint selection within the frozen budget remains validation-only per seed. Each graph-temporal head uses its own seed GAT encoder, frozen after that encoder’s validation selection. RF–AE fusion uses matching seed-specific predictions and recalibrated component thresholds. Numeric development thresholds are not copied across seeds.\n\n'
    summary+='Self-only is planned for all five seeds, subject only to documented resource feasibility. L4, alternative graph windows, IF retraining and weighted/MAX fusion are excluded from final fitting. This roster is recorded before additional seeds. Original Phase9 coverage-gate revision remains a development limitation; exact cohort/history identities and masking are now fixed, with no further gate relaxation.\n\n'
    summary+='## Dataset/cohort separation\n\nFull-data TRAIN/validation/test contain12,795,136 /1,652,943 /1,374,143 rows. Phase5 matched targets contain99,962 /206,603 /171,753. Feb20 graph and graph-temporal targets contain172,984 /88,782 /212,591. All IDs and order remain fixed. Do not compare metrics from different cohorts as component ablations. Full-data baseline predictions are sliced and thresholds recalibrated on the Phase5 matched validation cohort for temporal comparisons. Graph controls are independently fitted under the Feb20 TRAIN-only preprocessing, never reused from full-data Phase4 fits.\n\n'
    summary+='Baseline cleaning and preprocessing are immutable; full-data preprocessing used all TRAIN classes. AE fitting alone is benign-only. Full-data split dates and every feature, category, imputation and scaling value are embedded below. The graph start-time split is before10:30/[10:30,11:00)/from11:00 on Feb20, with completion-causal graph and temporal history and the original split-boundary limitations. Phase5 uses start-ordered global flow sequences; do not relabel it as completion-causal deployment replay.\n\n'
    summary+='## Thresholds, metrics and statistical interpretation\n\n'+METRICS.removeprefix('# Frozen metric definitions\n\n')+'\n'
    summary+='## Frozen research questions\n\n'+rq.removeprefix('# Frozen research questions\n\n')+'\n'
    summary+='## Frozen interpretation\n\n'+claims.removeprefix('# Claim freeze before additional seeds\n\n')+'\n'
    summary+='## Execution order, runtime and approval gate\n\n'+RUN_PLAN.removeprefix('# Final run plan — not executed\n\n')+'\n'
    summary+='Runtime is CPU-only on the recorded i7-8565U/16.94GB machine, two math/PyTorch threads and RF n_jobs1, one fitting process at a time. The machine/software and stage definitions are fully embedded below. Distinguish fitting from checkpoint selection and preparation, model compute from complete inference, and head size from shared-encoder pipeline size. Explicitly include separately measured history-index preparation; no unqualified comparisons across full-flow, sampled temporal and graph cohorts.\n\n'
    summary+='## Complete machine-readable configuration embedded for reproduction\n\nThe following is the complete JSON specification, not a selection based on test performance. “s” denotes the current seed. Source paths and SHA-256 values identify the immutable data/code required by these algorithms. Historical fields in provenance snapshots (including a pre-split proposal marked approved=false and old development method lists) describe earlier artifacts only: the explicit approved split record and final model registry/normalized protocols govern final execution. No new search is authorized by a historical field.\n\n```json\n'+json.dumps(spec,indent=2,sort_keys=True,allow_nan=False)+'\n```\n'
    (OUT/'FINAL_EXPERIMENT_SPEC.md').write_text(summary,encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    require(not (OUT/'final_experiment_spec.json').exists(),'Specification already frozen; use a separately approved version rather than overwrite')
    manifest=verify_development()
    spec,registry=make_spec()
    spec['artifact_hash_manifest']=artifact('results/final_spec_v1/artifact_hash_manifest.csv')
    sourcefiles=list((ROOT/'src/final_spec').glob('*.py'))
    spec['specification_generator']=[artifact(relative(p)) for p in sourcefiles]
    dump(OUT/'final_experiment_spec.json',spec)
    write_csv('model_registry.csv',registry)
    write_csv('seed_registry.csv',[dict(seed=s,development_evidence_exists=s==42,final_rerun_required=True,
        final_status='not_started_awaiting_explicit_approval',phase10_training_performed=False,
        required_stochastic_configurations=11,planned_resource_conditional_configurations=1,configuration_change_allowed=False) for s in SEEDS])
    document(spec,registry)
    # Rehash all protected files after specification construction, not just their metadata.
    for i,row in enumerate(manifest):
        require(sha(ROOT/row['path'])==row['sha256'],'Prior artifact modified during Phase10: '+row['path'])
        if i%150==0:print('Post-freeze protected recheck:',i+1,flush=True)
    count=sum(r['verification']=='historical_match' for r in manifest)
    report=f'''# Phase10 integrity report

Status: PASS — specification frozen, final execution awaiting explicit approval.

- {len(manifest)} protected artifacts hashed before and after document generation; no changes.
- {count} artifacts matched existing historical SHA-256 references. Remaining entries are explicitly labeled phase10_snapshot; their first freeze is not misrepresented as a historical verification.
- baseline_v1, Phase4/4A, Phase5, Phase6, Phase7, Phase8 and Phase9 protected, including checkpoint/score/cohort/config/source records and referenced raw/cleaned/cache files.
- No training module imported by this generator; no estimator fit, optimizer step, prediction recomputation or seed execution performed.
- All final seeds, methods, architectures, optimization budgets, validation-only calibration rules, data identities, dependency rules and exclusions frozen in final_experiment_spec.json.
- Full preprocessing states, feature order, partition definitions and protocols embedded in FINAL_EXPERIMENT_SPEC.md.
- Seed42 must be freshly rerun after approval; seeds123/456/789/1024 remain not started.
- Existing development scripts remain unchanged and are not final runners. Runtime versions must be verified before any future execution; no package installation or environment modification occurred in Phase10.

Specification SHA-256: `{sha(OUT/'final_experiment_spec.json')}`.
Artifact manifest SHA-256: `{sha(OUT/'artifact_hash_manifest.csv')}`.
'''
    (OUT/'integrity_report.md').write_text(report,encoding='utf-8')
    dump(OUT/'freeze_integrity.json',dict(status='PASS',protected_files=len(manifest),historical_hash_matches=count,
        prior_artifacts_unchanged=True,training_performed=False,execution_authorized=False,
        final_output_hashes={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='freeze_integrity.json'}))
    print('Phase10 specification frozen. No training executed. Await explicit execution approval.',flush=True)

if __name__=='__main__':main()
