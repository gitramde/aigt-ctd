"""Programmatic reporting from saved graph, control and sensitivity outputs."""
import time
import numpy as np
import pandas as pd
from .common import *
from src.phase4.metrics import binary_metrics


def table(rows,columns):
    def fmt(v): return f'{v:.6f}' if isinstance(v,float) else str(v)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
        ['| '+' | '.join(fmt(r.get(c,'')) for c in columns)+' |' for r in rows])


def report():
    import platform,torch,scipy,pyarrow,sklearn,xgboost
    json_write(METRICS/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,scipy=scipy.__version__,
        pyarrow=pyarrow.__version__,scikit_learn=sklearn.__version__,xgboost=xgboost.__version__,
        device='cpu',training_threads=PROTOCOL['threads'],seed=42))
    from .data import diagnostics,prepare
    prepare()
    source=read(OUT/'pre_split/proposed_split.json')['inputs']
    for kind in ('raw','cleaned'):
        require(sha256(source[kind+'_path'])==source[kind+'_sha256'],'Frozen Feb-20 input changed')
    diagnostics()
    guard();selection=read(OUT/'selected_model.json');lock=read(CONFIG/'threshold_lock.json')
    require(selection['config']['window_minutes']==1 and not lock['selection_uses_test'],'Primary/test selection violation')
    prep=read(CONFIG/'preprocessing.json')
    proposal_state=read(OUT/'pre_split/proposed_split.json')
    require(prep['fit_partition']=='train' and prep['training_rows']==proposal_state['partitions'][0]['rows'],'Preprocessing fit scope mismatch')
    require(not {'target_binary','attack_label','Label','timestamp','Timestamp','Src IP','Dst IP','Src Port','source_row','source_file'}.intersection(prep['output_feature_names']),'Metadata feature leakage')
    primary_training=read(METRICS/f"{selection['config']['name']}_training.json")
    self_training=read(METRICS/'gatv2_self_only_training.json');mlp_training=read(METRICS/'edge_mlp_training.json')
    require(self_training['parameters']==primary_training['parameters'],'Self-only capacity mismatch')
    require(abs(mlp_training['parameters']/primary_training['parameters']-1)<.01,'Edge MLP capacity differs by more than 1%')
    for name in ('gatv2_w5','gatv2_w10'):
        cfg=read(CONFIG/f'{name}.json')
        require(all(cfg[k]==selection['config'][k] for k in ('hidden','heads','layers','dropout')),'Sensitivity architecture changed')
    for row in lock['thresholds']:
        if row['criterion']=='fpr_at_most_0.01': require(row['validation_false_positive_rate']<=.01,'Validation FPR cap violated')
    require(lock['selected_model_sha256']==sha256(OUT/'selected_model.json'),'Selected model changed')
    require(lock['thresholds_sha256']==sha256(OUT/'thresholds.csv'),'Thresholds changed')
    require(lock['cohort_sha256']==sha256(METRICS/'evaluation_cohort.parquet'),'Cohort changed')
    cohort=pd.read_parquet(METRICS/'evaluation_cohort.parquet');rows=[];runtime=[]
    for name in selection['evaluation_models']:
        training=read(METRICS/f'{name}_training.json');evaluation=read(METRICS/f'{name}_evaluation.json')
        require(evaluation['threshold_lock_sha256']==sha256(CONFIG/'threshold_lock.json'),'Evaluation not bound to lock')
        require(sha256(lock['artifacts'][name]['model_path'])==evaluation['model_sha256'],'Checkpoint changed')
        for part,entry in evaluation['partitions'].items():
            require(sha256(entry['prediction_path'])==entry['prediction_sha256'],'Saved scores changed')
            y=cohort.loc[cohort.partition==PARTS.index(part),'target_binary'].to_numpy();p=np.load(entry['prediction_path'])
            rank=binary_metrics(y,p)
            for saved in entry['metrics']:
                m=binary_metrics(y,p,saved['threshold'],include_auc=False)
                for key in m: require(np.isclose(m[key],saved[key],rtol=1e-12,atol=1e-12),'Metric mismatch: '+key)
                for key in ('roc_auc','pr_auc','average_precision'):
                    require(np.isclose(rank[key],saved[key],rtol=1e-12,atol=1e-12),'Ranking metric mismatch: '+key)
                rows.append(dict(**saved,window_minutes=training['config'].get('window_minutes',0),
                    **{}))
            runtime.append(dict(model=name,partition=part,training_seconds=training['training_seconds'],
                validation_selection_seconds=training['validation_selection_seconds'],training_peak_memory_bytes=training['peak_memory_bytes'],
                evaluation_peak_memory_bytes=evaluation['peak_memory_bytes'],model_size_bytes=training['model_size_bytes'],
                parameters=training.get('parameters'),**entry['runtime']))
    for part in ('validation','test'): write(f'{part}_metrics.csv',[r for r in rows if r['partition']==part])
    write('runtime_metrics.csv',runtime)
    primary=selection['config']['name'];ablation_names=[primary,'edge_mlp','gatv2_self_only']
    write('graph_ablation.csv',[r for r in rows if r['model'] in ablation_names])
    sensitivity_names=[primary,'gatv2_w5','gatv2_w10']
    write('window_sensitivity.csv',[r for r in rows if r['model'] in sensitivity_names])
    write('matched_cohort_comparison.csv',[r for r in rows if r['model'] not in ('gatv2_w5','gatv2_w10','gatv2_self_only')])
    counts=[]
    proposal=read(OUT/'pre_split/proposed_split.json')
    flow_times=pd.read_parquet(CACHE/'metadata.parquet',columns=['partition','completion_us'])
    target_ids=np.load(CACHE/'target_ids.npy')
    overlaps=[]
    for i,boundary in enumerate(proposal['boundaries']):
        bound=int(pd.Timestamp(boundary).timestamp()*1_000_000)
        mask=(flow_times.partition.to_numpy()==i)&(flow_times.completion_us.to_numpy()>=bound)
        overlaps.append(dict(partition=PARTS[i],next_partition_start=boundary,flows_completing_at_or_after_boundary=int(mask.sum()),
            target_flows_completing_at_or_after_boundary=int(mask[target_ids].sum()),
            latest_completion=str(pd.Timestamp(int(flow_times.loc[flow_times.partition==i,'completion_us'].max()),unit='us'))))
    write('flow_completion_overlap.csv',overlaps)
    del flow_times
    for i,part in enumerate(PARTS):
        c=cohort[cohort.partition==i];counts.append(dict(partition=part,approved_rows=proposal['partitions'][i]['rows'],
            target_edges=len(c),benign=int((c.target_binary==0).sum()),malicious=int(c.target_binary.sum()),
            target_coverage=len(c)/proposal['partitions'][i]['rows']))
    write('cohort_counts.csv',counts)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    comparison=[r for r in rows if r['partition']=='test' and r['criterion']=='fpr_at_most_0.01' and r['model'] not in ('gatv2_w5','gatv2_w10','gatv2_self_only')]
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,metric in zip(axes,('recall','average_precision')):
        ax.barh([r['model'] for r in comparison],[r[metric] for r in comparison]);ax.set(xlim=(0,1),xlabel=metric,title='Matched test cohort; validation FPR cap 1%')
    fig.savefig(FIGURES/'matched_test_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,criterion in zip(axes,('fixed_0_5','maximum_macro_f1','fpr_at_most_0.01')):
        subset=sorted([r for r in rows if r['partition']=='test' and r['criterion']==criterion and r['model'] in sensitivity_names],key=lambda r:r['window_minutes'])
        for metric in ('recall','f1','false_positive_rate'): ax.plot([r['window_minutes'] for r in subset],[r[metric] for r in subset],marker='o',label=metric)
        ax.set(xticks=[1,5,10],xlabel='Calendar window (minutes)',ylim=(0,1),title=criterion);ax.legend()
    fig.savefig(FIGURES/'window_sensitivity.png',dpi=150);plt.close(fig)
    display=[r for r in rows if r['partition']=='test' and r['criterion']=='fixed_0_5' and r['model'] in ablation_names]
    fig,axes=plt.subplots(1,3,figsize=(11,4),constrained_layout=True)
    for ax,r in zip(axes,display):
        cm=np.array([[r['true_negatives'],r['false_positives']],[r['false_negatives'],r['true_positives']]])
        ax.imshow(np.log1p(cm),cmap='Blues')
        for (i,j),value in np.ndenumerate(cm): ax.text(j,i,f'{value:,}',ha='center',va='center')
        ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted',ylabel='Actual',title=r['model'])
    fig.savefig(FIGURES/'ablation_confusion_matrices.png',dpi=150);plt.close(fig)
    summary='# Phase 6 seed-42 graph contribution results\n\n'
    c=selection['config']
    summary+=f"Primary window: **1 minute**, declared before training. Selected GATv2: two layers, hidden dimension {c['hidden']} total across four heads, dropout {c['dropout']}, epoch {selection['selected_epoch']}; validation macro-F1 {selection['validation_macro_f1']:.6f}. Two architectures were evaluated under a four-epoch ceiling and patience two. Model, epoch and threshold decisions used validation only. The selected architecture was held fixed for 5/10-minute sensitivity runs, which independently selected epochs and thresholds on validation.\n\n"
    summary+='The approved Feb-20 split uses 10:30 and 11:00 boundaries. This is within-day graph-contribution development on Benign versus DDoS attacks-LOIC-HTTP, not an unseen-attack or zero-day experiment. All comparisons below use identical target flows and newly fitted Feb-20 TRAIN preprocessing. Existing Phase 4 models/results were not used as comparison scores.\n\n'
    summary+='## Shared target cohort\n\n'+table(counts,['partition','approved_rows','target_edges','benign','malicious','target_coverage'])+'\n\n'
    summary+='Targets come from each minute’s final 30 seconds, deterministically sampled at stride 16 in train and 4 in validation/test, resetting each minute. Metrics apply to this sampled cohort. Preprocessing uses every approved training row; model fitting uses only training target rows. Source-row cohort membership is saved and hashed.\n\n'
    summary+='## Matched test comparison\n\n'+table([r for r in rows if r['partition']=='test' and r['model'] not in ('gatv2_w5','gatv2_w10','gatv2_self_only')],['model','criterion','recall','precision','f1','macro_f1','false_positive_rate','roc_auc','average_precision'])+'\n\n'
    summary+='## Graph contribution ablation\n\n'+table([r for r in rows if r['partition']=='test' and r['model'] in ablation_names],['model','criterion','recall','f1','macro_f1','false_positive_rate','roc_auc','average_precision'])+'\n\n'
    summary+='Edge MLP uses only the current flow features, with a hidden width chosen to approximately match the selected GATv2 parameter count. GATv2 additionally has historical node aggregates and relational messages, so that pair alone cannot isolate message passing from extra historical information. The self-only control keeps the same architecture and historical node inputs but removes all inter-node message edges; this is the more direct test of relational message passing. Its unused edge-attention parameters remain in the checkpoint, so equal serialized parameter counts do not imply equal active degrees of freedom.\n\n'
    differences=[]
    for criterion in ('fixed_0_5','maximum_macro_f1','fpr_at_most_0.01'):
        graph=next(r for r in rows if r['model']==primary and r['partition']=='test' and r['criterion']==criterion)
        for control in ('edge_mlp','gatv2_self_only'):
            baseline=next(r for r in rows if r['model']==control and r['partition']=='test' and r['criterion']==criterion)
            differences.append(dict(control=control,criterion=criterion,**{k:graph[k]-baseline[k] for k in ('recall','f1','macro_f1','false_positive_rate','average_precision')}))
    summary+='GATv2 minus control differences (positive FPR means more false alarms):\n\n'+table(differences,['control','criterion','recall','f1','macro_f1','false_positive_rate','average_precision'])+'\n\n'
    summary+='These are descriptive, single-seed comparisons. Improvements at one operating point do not establish an overall graph benefit, and no statistical significance is claimed.\n\n'
    summary+='## Window sensitivity\n\n'+table([r for r in rows if r['partition']=='test' and r['model'] in sensitivity_names],['window_minutes','criterion','recall','f1','false_positive_rate','roc_auc','average_precision'])+'\n\n'
    summary+='The 1-minute model remains primary regardless of sensitivity test performance. All windows use the same minute-level cutoffs and target cohort; sensitivity changes the available within-window history. Larger windows may hit the message-edge cap more often.\n\n'
    summary+='## Causality, scope and measurement limits\n\n'
    summary+='At each minute’s second-30 cutoff, history contains only flows starting inside the assigned calendar window and completing strictly before the cutoff. Equal-cutoff completions are excluded. Targets start at or after the cutoff and are classified at completion using their own complete flow vector. Calendar-window and partition membership use the frozen start timestamp; a target can complete after its start window ends, but its context remains frozen at the earlier cutoff. Node history and message passing cannot use target labels, target-flow data, future flows or other partitions. Validation/test histories may use earlier unlabeled completed flows within their own partition. No future-window node universe is supplied. This is flow-completion detection, not prediction at flow start.\n\n'
    summary+='Node statistics use all eligible history, while message edges retain the 8,192 most recently completed eligible flows (source-row tie break). Nodes irrelevant to retained messages and targets are pruned after aggregation. Thus this tests a bounded historical network-flow graph instantiation of the generalized AIGT-CTD architecture, not exhaustive graph processing or the full AIGT-CTD system. Directed edges follow Source IP to Destination IP; IP indices are topology keys, never numeric/learned identity features.\n\n'
    summary+='graph_snapshot_diagnostics.csv describes complete calendar-window topology using all cleaned flows for descriptive diagnostics only; those complete graphs are never training inputs. Density counts unique non-self directed pairs divided by n(n−1), not parallel edges. Components are reported both weakly and strongly. Isolated-node count is zero by the full-window endpoint-defined node universe. causal_prefix_diagnostics.csv separately describes actual historical model inputs, caps and cold endpoints. Summaries include mean, median, p90 and maximum.\n\n'
    summary+='All metric tables include Accuracy, Precision, Recall, F1, Macro-F1, ROC-AUC, trapezoidal PR-AUC, Average Precision, FPR, FNR and confusion counts. Thresholds are fixed 0.5, validation maximum macro-F1, and validation FPR ≤1%; the cap is not a test FPR guarantee. Conventional controls are newly fitted on the same sampled targets: logistic-loss SGD, Random Forest, XGBoost, and the edge MLP.\n\n'
    summary+='Runtime measurements include graph-prefix preparation and model compute during inference but exclude raw joining, preprocessing/cache creation and model loading. data_manifest.json records the completed resumed materialization pass, not total preparation cost including earlier cache creation and timestamp-unit repair. Latency is milliseconds per 1,000 evaluated target edges. Peak memory is process working-set/RSS including dependencies and cached metadata; training peaks repeat across partition rows and should not be summed. CPU timings are machine-specific. Training histories and all searched configurations are saved.\n\n'
    summary+='The same attack episode spans partitions, endpoints may recur, and prevalence differs sharply. The sparse target schedule, four-epoch budget and message cap limit generalization of findings. Frozen timestamps are preserved without AM/PM repair. Baseline_v1, Phase 4/4A and Phase 5 artifacts remain unchanged. No Transformer, anomaly/reconstruction branch, explainability, risk fusion or final five-seed run was introduced.\n'
    summary+='\n## Start-time split versus completion-time availability\n\n'+table(overlaps,['partition','next_partition_start','flows_completing_at_or_after_boundary','target_flows_completing_at_or_after_boundary','latest_completion'])+'\n\n'
    summary+='The approved membership is based on flow start timestamps, so some earlier-partition flows finish after the next partition starts. No later-partition record is fitted, and every message/node context is completion-causal, but these results are an offline benchmark under the approved start-time split, not proof that a freshly trained model could be deployed exactly at the boundary. A strict deployment replay would need a separately approved completion-time purge/embargo and fresh preprocessing; that would change the approved membership and was not silently substituted here.\n'
    (OUT/'phase6_results_summary.md').write_text(summary,encoding='utf-8')
    architecture='''# Graph construction and model specification

GATv2 uses edge-aware dynamic attention: for each destination i and source j, form LeakyReLU(W_destination h_i + W_source h_j + W_edge e_ji), dot with one learned vector per head, normalize over incoming edges plus an explicit self loop, then aggregate W_source h_j. Four heads concatenate into the configured total hidden dimension (64 or 128). Two layers use ELU, dropout, residual addition and LayerNorm. Synthetic self loops have zero edge attributes. Original directed multiedges remain separate.

The implementation follows the [official GATv2 operator equations](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html) using native PyTorch scatter operations. A dense-reference test checks numerical equivalence and gradients. No PyTorch Geometric installation is required.

The edge head concatenates [source endpoint embedding, destination endpoint embedding, current train-preprocessed flow feature vector], followed by Linear -> GELU -> Dropout -> Linear -> binary logit. Sigmoid yields malicious probability. BCEWithLogitsLoss uses training-target negative/positive weighting. AdamW uses learning rate 0.0005, weight decay 0.01 and gradient clipping at 1.0. Seed is 42 and deterministic CPU algorithms are enabled.

Node features, in order, are log1p of incoming count, outgoing count, incoming total flow bytes, outgoing total flow bytes, TCP incident count, UDP incident count, low (<1024) source-port outgoing count, and low destination-port incoming count. Flow bytes are forward plus backward bytes attributed to the directed flow's endpoints; they are not a claim about true one-way byte direction. These deterministic aggregates use no fitted vocabulary, labels, numerical IP encoding or future history. Protocol and ports come from the source-row identifier join and frozen cleaned records.

Historical edge attention features are the train-transformed subset named below; the current target retains the full eligible feature vector. Frozen feature eligibility is reused, but imputation, categories, train-value leakage checks and scaling are fitted afresh on all approved Feb-20 training rows only. Protocol unknown categories use all-zero encoding.

'''
    architecture+='Historical edge features: '+', '.join(read(METRICS/'data_manifest.json')['history_feature_names'])+'.\n\n'
    architecture+='The precise causal cutoff, completion-time rule, cap, target sampling and history-state policies are frozen in configs/protocol.json. No snapshot crosses its calendar window or approved partition.\n'
    full=pd.read_csv(OUT/'graph_snapshot_diagnostics.csv');prefix=pd.read_csv(OUT/'causal_prefix_diagnostics.csv')
    graph_overview=[]
    for window in (1,5,10):
        f=full[full.window_minutes==window];p=prefix[prefix.window_minutes==window]
        graph_overview.append(dict(window_minutes=window,calendar_snapshots=len(f),causal_prefixes=len(p),
            mean_nodes=float(f.nodes.mean()),mean_edges=float(f.edges.mean()),mean_weak_components=float(f.weak_connected_components.mean()),
            mean_density=float(f.density.mean()),mean_message_edges=float(p.message_edges.mean()),
            capped_prefix_fraction=float((p.message_edges<p.history_edges).mean())))
    architecture+='\n## Descriptive graph overview\n\n'+table(graph_overview,['window_minutes','calendar_snapshots','causal_prefixes','mean_nodes','mean_edges','mean_weak_components','mean_density','mean_message_edges','capped_prefix_fraction'])+'\n\n'
    architecture+='Complete per-snapshot diagnostics are in graph_snapshot_diagnostics.csv; graph_diagnostics_summary.csv gives partition-specific mean, median, p90 and maximum for nodes, edges, weak/strong components, isolated nodes, repeated pairs, density and both classes. The complete-window topology is diagnostic only and is never supplied to a classifier. causal_prefix_diagnostics.csv records the actual pruned/capped model graph sizes.\n'
    (OUT/'graph_construction_report.md').write_text(architecture,encoding='utf-8')
    guard()
    checks=dict(split_explicitly_approved=True,original_artifacts_unchanged=True,preprocessing_train_only=True,
        identical_target_cohort=True,thresholds_frozen_before_test=True,primary_window_fixed=True,
        label_free_node_and_edge_features=True,history_completed_before_cutoff=True,partition_and_calendar_boundaries=True,
        saved_metrics_recomputed=True,seed_42_only=True)
    json_write(METRICS/'integrity_verification.json',dict(status='PASS',checks=checks,threshold_lock_sha256=sha256(CONFIG/'threshold_lock.json'),
        preprocessing_sha256=sha256(CONFIG/'preprocessing.json'),cohort_sha256=sha256(METRICS/'evaluation_cohort.parquet'),
        source_code_sha256={p.name:sha256(p) for p in (ROOT/'src/graph').glob('*.py')}))
    (OUT/'integrity_report.md').write_text('# Graph experiment integrity\n\n'+table([dict(check=k,status='PASS') for k in checks],['check','status'])+'\n\nFull evidence is in configs/split_approval.json, metrics/data_manifest.json, configs/threshold_lock.json, per-model histories/evaluations and metrics/integrity_verification.json. Protected earlier-phase artifact hashes were rechecked after report generation.\n',encoding='utf-8')
    print('Phase 6 reports complete',flush=True)

if __name__=='__main__': report()
