"""Lock validation thresholds, then evaluate matched controls and temporal heads."""
from .train import *
from src.phase4a.run import select_thresholds
from .report import table,protected_check,coverage_text,LIMITS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    require(read(OUT/'metrics/coverage_gate.json')['passed'],'Coverage gate failed')
    seed();y=np.load(OUT/'metrics/targets_y.npy');part=np.load(OUT/'metrics/target_partitions.npy')
    names=['edge_mlp',NAME]+[f'gat_transformer_L{l}' for l in (1,4,8)]
    points=[];probabilities={};search=[];training={}
    for name in names:
        root=GRAPH if name in names[:2] else OUT
        record=read(root/f'metrics/{name}_training.json');training[name]=record
        path=root/f'metrics/{name}_validation.npy'
        require(sha256(path)==record['validation_prediction_sha256'],'Validation scores changed')
        p=np.load(path);require(len(p)==int((part==1).sum()),'Control validation cohort mismatch')
        probabilities[(name,'validation')]=p
        thresholds=[dict(criterion='fixed_0_5',threshold=.5)]+[r for r in select_thresholds(y[part==1],p) if r['criterion'] in ('maximum_macro_f1','fpr_at_most_0.01')]
        points.extend(dict(model=name,criterion=r['criterion'],threshold=r['threshold'],selection_partition='validation') for r in thresholds)
        search.append(dict(model=name,length=record.get('length',''),fitted_in_phase9=name not in names[:2],
            selected_epoch=record['selected_epoch'],executed_epochs=record['executed_epochs'],
            selected_validation_macro_f1=record['selected_validation_macro_f1']))
    csv_write(OUT/'development_search.csv',search);csv_write(OUT/'thresholds.csv',points)
    json_write(OUT/'configs/threshold_lock.json',dict(selection_uses_test=False,all_lengths_reported=True,thresholds=points,
        frozen_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),threshold_csv_sha256=sha256(OUT/'thresholds.csv'),
        cohort_manifest_sha256=sha256(OUT/'matched_cohort_manifest.json'),
        validation_predictions={name:sha256((GRAPH if name in names[:2] else OUT)/f'metrics/{name}_validation.npy') for name in names}))
    print('All thresholds locked before temporal test inference',flush=True)
    representations=np.load(OUT/'metrics/representations.npy',mmap_mode='r');history=np.load(OUT/'metrics/history_indices.npy')
    rep=read(OUT/'metrics/representation_manifest.json');runtime=[];metrics=[]
    for name in names:
        record=training[name]
        with MemoryMonitor() as memory:
            if name not in names[:2]:
                model=TemporalHead(representations.shape[1]);model.load_state_dict(torch.load(OUT/f'models/{name}.pt',weights_only=True,map_location='cpu'))
            for partition in ('validation','test'):
                mask=part==PARTS.index(partition)
                if name in names[:2]:
                    evaluation=read(GRAPH/f'metrics/{name}_evaluation.json');old=evaluation['partitions'][partition]
                    require(sha256(old['prediction_path'])==old['prediction_sha256'],'Control prediction mismatch')
                    p=np.load(old['prediction_path']);elapsed=old['runtime']['inference_seconds'];rep_seconds=0.
                    peak=evaluation['peak_memory_bytes'];runtime_scope='Phase6 original measured full matched cohort; predictions reused; no refit'
                else:
                    p,elapsed=predict(model,representations,history,np.flatnonzero(mask),record['length'])
                    if partition=='validation':
                        require(np.allclose(p,probabilities[(name,partition)],rtol=1e-6,atol=1e-7),'Validation replay mismatch')
                        p=probabilities[(name,partition)]
                    rep_seconds=next(r['representation_seconds'] for r in rep['timings'] if r['partition']==partition)
                    peak=max(memory.peak,rep['peak_memory_bytes']);runtime_scope='amortized frozen graph preparation+encoding plus cached temporal lookup/head; excludes metadata/model load and one-time history-index preparation'
                require(len(p)==mask.sum(),'Matched cohort mismatch');np.save(OUT/f'metrics/{name}_{partition}.npy',p)
                rank=binary_metrics(y[mask],p)
                for setting in [r for r in points if r['model']==name]:
                    row=dict(model=name,partition=partition,criterion=setting['criterion'],**binary_metrics(y[mask],p,setting['threshold'],include_auc=False),
                             **{k:rank[k] for k in ('roc_auc','pr_auc','average_precision')})
                    metrics.append(row)
                root=GRAPH if name in names[:2] else OUT
                serialized=record['model_size_bytes']
                encoder_bytes=0 if name in names[:2] else (GRAPH/f'models/{NAME}.pt').stat().st_size
                runtime.append(dict(model=name,partition=partition,training_seconds=record['training_seconds'],
                    phase9_training_seconds=record['training_seconds'] if name not in names[:2] else 0,
                    inference_seconds=elapsed+rep_seconds,temporal_head_and_lookup_seconds=elapsed if name not in names[:2] else 0,
                    frozen_graph_representation_seconds=rep_seconds,latency_ms_per_1000=(elapsed+rep_seconds)/len(p)*1e6,
                    peak_memory_bytes=peak,training_peak_memory_bytes=record['peak_memory_bytes'],model_size_bytes=serialized,
                    shared_frozen_encoder_bytes=encoder_bytes,total_required_model_bytes=serialized+encoder_bytes,runtime_scope=runtime_scope))
                print('Evaluated',name,partition,flush=True)
        if name not in names[:2]:
            for row in runtime:
                if row['model']==name:row['peak_memory_bytes']=max(memory.peak,rep['peak_memory_bytes'])
    for partition in ('validation','test'):csv_write(OUT/f'{partition}_metrics.csv',[r for r in metrics if r['partition']==partition])
    csv_write(OUT/'component_progression.csv',metrics);csv_write(OUT/'runtime_metrics.csv',runtime)
    differences=[]
    for partition in ('validation','test'):
        for length in (4,8):
            for criterion in ('fixed_0_5','maximum_macro_f1','fpr_at_most_0.01'):
                control=next(r for r in metrics if r['model']=='gat_transformer_L1' and r['partition']==partition and r['criterion']==criterion)
                modelrow=next(r for r in metrics if r['model']==f'gat_transformer_L{length}' and r['partition']==partition and r['criterion']==criterion)
                differences.append(dict(partition=partition,model=modelrow['model'],control='gat_transformer_L1',criterion=criterion,
                    **{f'delta_{k}':modelrow[k]-control[k] for k in ('recall','precision','f1','macro_f1','false_positive_rate','roc_auc','average_precision')}))
    csv_write(OUT/'temporal_ablation.csv',differences)
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    for ax,criterion in zip(axes,('fixed_0_5','maximum_macro_f1','fpr_at_most_0.01')):
        subset=[r for r in metrics if r['partition']=='test' and r['criterion']==criterion]
        ax.barh([r['model'] for r in subset],[r['macro_f1'] for r in subset]);ax.set(xlabel='Macro-F1',title=criterion,xlim=(0,1))
    fig.savefig(OUT/'figures/component_progression.png',dpi=150);plt.close(fig)
    count=protected_check()
    summary='# Phase 9 seed-42 graph + temporal contribution results\n\n'
    summary+='Frozen Phase-6 primary GATv2 encoder (1-minute window, hidden 128, two layers, four heads, dropout 0.2), followed by separately fitted identical Transformer heads (d_model 64, two layers, four heads, FFN 128, dropout 0.1). Only Transformer heads are trained; this evaluates temporal readout of frozen graph-derived representations, not end-to-end joint training. No graph architecture search was reopened.\n\n'
    summary+='All original target edges are retained; current-token availability permits masking absent history. Existing Edge MLP/GAT predictions are reused without refitting and metrics recomputed on exactly this cohort. L1/L4/L8 have equal trainable parameter counts and the same eight-slot position table; unused historical positions remain inactive at L1.\n\n'
    summary+=coverage_text()
    summary+='## Training and selection\n\nAdamW lr 0.0005, weight decay 0.01, TRAIN-only positive class weight, batch size 2048, clipping 1.0, at most four epochs and patience two. Epoch selection uses validation macro-F1 at 0.5 only. All sequence lengths are reported; no test-selected winner. Learned positions encode the fixed minute slots; the current output attends only to current and completion-eligible past tokens. Frozen embedding extraction uses no labels.\n\n'+table(search,list(search[0]))+'\n\n'
    test=[r for r in metrics if r['partition']=='test']
    summary+='## Matched test component progression\n\n'+table(test,['model','criterion','recall','precision','f1','macro_f1','false_positive_rate','roc_auc','average_precision'])+'\n\n'
    summary+='## Temporal contribution: temporal minus L1 capacity control\n\nPositive FPR differences mean more false alarms. Differences are absolute fractions.\n\n'+table([r for r in differences if r['partition']=='test'],['model','criterion','delta_recall','delta_precision','delta_f1','delta_macro_f1','delta_false_positive_rate','delta_roc_auc','delta_average_precision'])+'\n\n'
    for length in (4,8):
        ds=[r for r in differences if r['partition']=='test' and r['model']==f'gat_transformer_L{length}']
        consistent=all(r['delta_macro_f1']>0 and r['delta_f1']>0 and r['delta_recall']>=0 and r['delta_false_positive_rate']<=0 for r in ds)
        summary+=f'L{length}: '+('descriptive gains are consistent under this conservative operating-point check, but do not establish statistical benefit.' if consistent else 'no consistent temporal benefit across the requested operating points under the joint recall/F1/macro-F1/FPR check.')+'\n\n'
    summary+='## Runtime and measurement scope\n\n'+table([r for r in runtime if r['partition']=='test'],['model','training_seconds','inference_seconds','latency_ms_per_1000','peak_memory_bytes','model_size_bytes','total_required_model_bytes'])+'\n\n'
    summary+='Temporal inference time is an amortized sum of frozen graph-prefix preparation/encoding and measured cached temporal lookup/head inference. Historical tokens are reused rather than recomputed for every target. Metadata/model loading and one-time history-index construction are excluded; representation timings are saved separately. These are not strict online-replay timings. Prior controls retain their original full-cohort Phase-6 timing, while Phase-9 training time for those controls is zero. Training timings for temporal models exclude the shared frozen encoder preparation. Peak memory is process high-water RSS including dependencies, with the maximum of separate representation and head stages; repeated training values must not be summed. Full model size includes the shared frozen encoder once per deployable pipeline; head checkpoint size is also reported.\n\n'
    summary+=LIMITS
    (OUT/'phase9_results_summary.md').write_text(summary,encoding='utf-8')
    (OUT/'integrity_report.md').write_text(f'# Phase 9 integrity\n\nPASS: Phase-6 hashes and approved membership; same target cohort for all models; completion-causal endpoint-consistent history; masked missing minutes; train-only class weight; validation-only epochs/thresholds; all lengths reported; seed42 only; {count} protected prior artifacts unchanged.\n\nEvidence: protected_artifacts.json, phase6_hash_verification.json, coverage_gate.json, matched_cohort_manifest.json, protocol.json, threshold_lock.json and per-model histories.\n',encoding='utf-8')
    json_write(OUT/'metrics/integrity_verification.json',dict(status='PASS',protected_files=count,
        source_sha256={p.name:sha256(p) for p in (ROOT/'src/graph_temporal').glob('*.py')}))
    print('Phase9 complete; stopping after seed42 development',flush=True)

if __name__=='__main__':main()
