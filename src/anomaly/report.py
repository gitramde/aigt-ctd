"""All Phase 7 tables and figures derive from immutable saved anomaly scores."""
import time,platform
import numpy as np
import pandas as pd
from .data import *
from .evaluate import LOCK_A,LOCK_B,names
from .metrics import metrics


def table(rows,cols):
    def fmt(x): return f'{x:.6f}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+
        ['| '+' | '.join(fmt(r.get(c,'')) for c in cols)+' |' for r in rows])


def score_summary(values):
    return dict(N=len(values),mean=float(np.mean(values)),median=float(np.median(values)),
        **{f'p{q}':float(np.percentile(values,q)) for q in (75,90,95,99)},maximum=float(np.max(values)))


def report():
    guard(full=True);a=read(LOCK_A);b=read(LOCK_B);selected=read(OUT/'selected_autoencoder.json')
    require(a['selected_autoencoder_sha256']==sha256(OUT/'selected_autoencoder.json'),'Model selection changed')
    require(not a['selection_uses_malicious_validation'] and not a['selection_uses_test'],'Protocol A contamination')
    require(not selected['selection_uses_malicious_validation'] and not selected['selection_uses_test'],'AE selection contamination')
    require(b['protocol_A_lock_sha256']==sha256(LOCK_A) and not b['selection_uses_test'],'Protocol B lock mismatch')
    require(a['csv_sha256']==sha256(OUT/'benign_only_thresholds.csv') and b['csv_sha256']==sha256(OUT/'label_aware_thresholds.csv'),'Threshold tables changed')
    training_manifest=read(OUT/'anomaly_training_manifest.json')
    for artifact in training_manifest['artifacts']: require(sha256(artifact['path'])==artifact['sha256'],'Anomaly fitting inputs changed')
    pre=read(BASE/'metrics/pretraining_check.json');train_counts=pre['attack_family_counts']['train']
    import pyarrow.parquet as pq
    for part in ('validation','test'):
        y,codes,book=labels(part);offset=0
        membership=read(BASE/'split_plan.json')['partitions'][part]['membership_path']
        for batch in pq.ParquetFile(membership).iter_batches(columns=['target_binary','attack_label'],batch_size=50000):
            stop=offset+len(batch)
            require(np.array_equal(y[offset:stop],batch.column(0).to_numpy()),'Diagnostic target cache differs from frozen membership')
            require(np.array_equal(np.asarray(book)[codes[offset:stop]],batch.column(1).to_numpy(zero_copy_only=False)),'Family cache differs from frozen membership')
            offset=stop
        require(offset==len(y),'Diagnostic cache membership incomplete')
    require(set(pre['attack_family_counts']['test'])-{'Benign'}=={'Bot','Infilteration'},'Unexpected test attack family')
    require(train_counts.get('Bot',0)==train_counts.get('Infilteration',0)==0,'Test attacks present in TRAIN')
    all_metrics=[];family_rows=[];summaries=[];runtime=[];comparison=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name in names():
        tr=read(METRICS/f'{name}_training.json');scoring=read(METRICS/f'{name}_scoring.json')
        require(sha256(tr['model_path'])==tr['model_sha256']==a['artifacts'][name]['model_sha256'],'Model changed')
        require(scoring['protocol_A_lock_sha256']==sha256(LOCK_A),'Scores not bound to primary lock')
        distributions={}
        for part in ('train_benign','validation','test'):
            entry=scoring['partitions'][part]
            require(sha256(entry['score_path'])==entry['score_sha256'],'Score artifact changed')
            scores=np.load(entry['score_path'],mmap_mode='r')
            runtime.append(dict(model=name,partition=part,training_seconds=tr['training_seconds'],benign_validation_selection_seconds=tr['benign_validation_selection_seconds'],
                training_peak_memory_bytes=tr['peak_memory_bytes'],scoring_peak_memory_bytes=scoring['peak_memory_bytes'],
                model_size_bytes=tr['model_size_bytes'],training_rows=tr['training_rows'],
                **{k:entry[k] for k in ('inference_seconds','prediction_only_seconds','latency_ms_per_1000','evaluated_records')}))
            if part=='train_benign': distributions['Train benign']=scores;continue
            y,codes,book=labels(part);require(len(y)==len(scores),'Full-cohort score count mismatch')
            rank=metrics(y,scores,0.)
            required=['Benign','Infilteration','Brute Force -Web','Brute Force -XSS','SQL Injection'] if part=='validation' else ['Bot','Infilteration']
            for label in (['Benign','Infilteration'] if part=='validation' else ['Benign','Infilteration','Bot']):
                distributions[part.title()+' '+label]=scores[codes==book.index(label)]
            for t in a['rows']+b['rows']:
                if t['model']!=name: continue
                row=dict(model=name,partition=part,protocol=t['protocol'],criterion=t['criterion'],seed=42,
                    **metrics(y,scores,t['threshold'],ranking=False),**{k:rank[k] for k in ('roc_auc','pr_auc','average_precision')})
                all_metrics.append(row)
                for family in required:
                    mask=codes==book.index(family);values=scores[mask];n=len(values);hits=int((values>=t['threshold']).sum())
                    family_rows.append(dict(model=name,partition=part,protocol=t['protocol'],criterion=t['criterion'],threshold=t['threshold'],
                        attack_family=family,training_status='KNOWN' if train_counts.get(family,0)>0 else 'UNSEEN',N=n,
                        detected=hits,missed=n-hits if family!='Benign' else None,recall=hits/n if n and family!='Benign' else None,
                        benign_false_positive_rate=hits/n if n and family=='Benign' else None))
                if part=='validation' and t['protocol']=='A_benign_only':
                    require(row['false_positive_rate']<=t['target_fpr']+1e-12,'Primary calibration FPR cap mismatch')
                if part=='test' and t['protocol']=='A_benign_only' and t['target_fpr']==.01:
                    family_recall={family:float((scores[codes==book.index(family)]>=t['threshold']).mean()) for family in ('Bot','Infilteration')}
                    comparison.append(dict(model=name,threshold_protocol='A: benign-validation-only empirical 1% FPR',threshold=t['threshold'],
                        actual_test_fpr=row['false_positive_rate'],aggregate_malicious_recall=row['recall'],precision=row['precision'],f1=row['f1'],AP=row['average_precision'],
                        Bot_recall=family_recall['Bot'],Infilteration_recall=family_recall['Infilteration'],records=len(y)))
        for group,values in distributions.items(): summaries.append(dict(model=name,group=group,**score_summary(values)))
        fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
        transformed={group:np.log10(1+values) for group,values in distributions.items()}
        bins=np.linspace(min(float(v.min()) for v in transformed.values()),max(float(v.max()) for v in transformed.values()),121)
        for group,values in transformed.items():
            hist,edges=np.histogram(values,bins=bins);ax.stairs(hist/len(values),edges,label=f'{group} (N={len(values):,})')
        ax.set(xlabel='log10(1 + raw anomaly score)',ylabel='Fraction of group per bin',title=name+' score distributions');ax.legend(fontsize=8)
        fig.savefig(FIGURES/f'{name}_score_histograms.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
        for group,values in transformed.items():
            ordered=np.sort(values);idx=np.unique(np.linspace(0,len(values)-1,min(3000,len(values))).astype(int))
            ax.plot(ordered[idx],(idx+1)/len(values),label=group)
        ax.set(xlabel='log10(1 + raw anomaly score)',ylabel='Empirical cumulative probability',title=name+' ECDF',ylim=(0,1));ax.legend(fontsize=8)
        fig.savefig(FIGURES/f'{name}_score_ecdf.png',dpi=150);plt.close(fig)
    baseline_lock=read(BASE/'diagnostics/validation_threshold_lock.json')
    require(not baseline_lock['selection_uses_test'],'Supervised baseline threshold provenance invalid')
    y,codes,book=labels('test')
    for name in ('logistic_regression','random_forest','xgboost','mlp'):
        t=next(r for r in baseline_lock['thresholds'] if r['model']==name and r['criterion']=='fpr_at_most_0.01')
        entry=read(BASE/'metrics'/f'{name}_seed42_evaluation.json')['partitions']['test']
        require(sha256(entry['probability_path'])==entry['probability_sha256'],'Phase 4 predictions changed')
        score=np.load(entry['probability_path'],mmap_mode='r');require(len(score)==len(y),'Supervised cohort mismatch')
        m=metrics(y,score,t['threshold'])
        comparison.append(dict(model=name,threshold_protocol='Phase 4A: label-aware validation recall-maximizing 1% FPR constraint',threshold=t['threshold'],
            actual_test_fpr=m['false_positive_rate'],aggregate_malicious_recall=m['recall'],precision=m['precision'],f1=m['f1'],AP=m['average_precision'],
            Bot_recall=float((score[codes==book.index('Bot')]>=t['threshold']).mean()),
            Infilteration_recall=float((score[codes==book.index('Infilteration')]>=t['threshold']).mean()),records=len(y)))
    write('anomaly_score_summary.csv',summaries);write('attack_family_diagnostics.csv',family_rows)
    write('runtime_metrics.csv',runtime);write('supervised_vs_anomaly_comparison.csv',comparison)
    for part in ('validation','test'): write(f'{part}_metrics.csv',[r for r in all_metrics if r['partition']==part])
    fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
    for ax,metric in zip(axes,('aggregate_malicious_recall','actual_test_fpr')):
        ax.barh([r['model'] for r in comparison],[r[metric] for r in comparison]);ax.set(xlabel=metric,title='Validation-calibrated operating points')
        if metric=='actual_test_fpr': ax.axvline(.01,color='red',linestyle='--',label='1% reference');ax.legend()
    fig.savefig(FIGURES/'supervised_vs_anomaly.png',dpi=150);plt.close(fig)
    summary='# Phase 7 seed-42 anomaly branch results\n\n'
    summary+=f"Selected autoencoder: **{selected['name']}**, hidden layers {selected['architecture']}, epoch {selected['selected_epoch']}, benign-validation MSE {selected['benign_validation_mse']:.9g}. Both candidate architectures visited all **{training_manifest['benign_train_rows']:,} benign TRAIN records in every executed epoch**, with no AE subsampling. Architecture and checkpoint selection minimized benign-validation reconstruction error; no malicious validation scores selected the representation.\n\n"
    summary+='The frozen baseline_v1 chronological split, preprocessing, feature definitions and labels were reused exactly. TRAIN spans Feb 14–22, validation Feb 23 and Feb 28, and test Mar 1–2. No Phase 4/4A/5/6 artifacts were modified. Seed 42 is development only; no statistical significance or final multi-seed result is claimed.\n\n'
    summary+='## Primary benign-only calibration (Protocol A)\n\nThresholds use only benign validation scores at empirical FPR caps 5%, 1%, 0.5% and 0.1%. Tied scores are handled conservatively using an upper order statistic and the next representable greater value, with detection defined as score ≥ threshold. Every Protocol-A threshold was frozen before malicious validation/test scoring. These are empirical validation caps, not a guarantee of test/population FPR.\n\n'
    summary+=table([r for r in all_metrics if r['partition']=='test' and r['protocol']=='A_benign_only'],['model','criterion','recall','precision','f1','macro_f1','false_positive_rate','false_negative_rate','roc_auc','pr_auc','average_precision'])+'\n\n'
    summary+='## Test attack families absent from TRAIN\n\n'+table([r for r in family_rows if r['partition']=='test' and r['protocol']=='A_benign_only'],['model','criterion','attack_family','N','detected','missed','recall'])+'\n\n'
    summary+='Bot and Infilteration are absent from the original TRAIN partition. They are reported as unseen attack families, not zero-day attacks. KNOWN/UNSEEN in validation diagnostics refers to presence in the full frozen TRAIN partition; the anomaly estimators themselves fit only benign rows. Benign diagnostic detection counts mean false alarms, with benign FPR reported instead of malicious-family recall.\n\n'
    summary+='## Label-aware calibration (Protocol B; comparison only)\n\nProtocol B selects maximum macro-F1 and maximum binary F1 from scored validation records and their labels. Infilteration appears in this calibration, so these are **not strict development-held-out Infilteration results**. Test labels never selected thresholds. Protocol B did not change the autoencoder architecture or model weights.\n\n'
    summary+=table([r for r in all_metrics if r['partition']=='test' and r['protocol']=='B_label_aware'],['model','criterion','recall','precision','f1','macro_f1','false_positive_rate','roc_auc','average_precision'])+'\n\n'
    summary+='## Comparison at validation-calibrated 1% FPR\n\n'+table(comparison,['model','actual_test_fpr','aggregate_malicious_recall','precision','f1','AP','Bot_recall','Infilteration_recall'])+'\n\n'
    summary+='Anomaly rows use **benign-only Protocol A**. Supervised rows reuse the existing **label-aware Phase 4A thresholds** and frozen Phase 4 predictions, selected to maximize validation recall under a 1% validation FPR constraint. These calibration protocols are different and should not be presented as equivalent. Every row evaluates exactly the full same test membership; Phase 4 was not retrained and Phase 5 sampled cohorts were not substituted. Actual test FPR, not the validation target alone, determines whether a method controls false alarms after temporal transfer.\n\n'
    summary+='## Score distributions and runtime\n\n'+table(summaries,['model','group','N','mean','median','p75','p90','p95','p99','maximum'])+'\n\n'
    summary+='Histogram and ECDF figures show log10(1 + raw score) for readability; metrics and thresholds always use raw scores. AE scores are per-record feature-mean reconstruction MSE; Isolation Forest uses negative score_samples so larger values mean greater anomaly. Scores are not malicious probabilities. PR-AUC is trapezoidal precision-recall area; AP is reported separately. Train-benign distributions use all benign TRAIN records for both models, even though Isolation Forest was fitted on a sample.\n\n'
    summary+='Benign calibration scores are retained exactly in the full validation score array after an exact checkpoint replay using the original batch layout. Small float32 reconstruction differences from mixed-partition batch layouts are checked and recorded in scoring manifests; they do not change the frozen thresholds.\n\n'
    isolation=read(METRICS/'isolation_forest_training.json')
    summary+=f"Isolation Forest fit on a deterministic seed-42 uniform sample of {isolation['training_rows']:,} benign TRAIN records without replacement. It uses 100 trees with 256 records per tree; {isolation['unique_rows_used_by_trees']:,} distinct sampled records participated across trees. Sampling limits memory and CPU use; sample IDs and hashes are saved. No validation/test row enters fitting.\n\n"
    summary+='Runtime tables separate training, benign-validation scoring/selection, inference, prediction-only compute and milliseconds per 1,000 scored records. End-to-end validation/test inference includes frozen transformation and checksum verification; training-benign scoring reads the existing verified feature cache. Peak memory is process high-water RSS/working set, and serialized size is checkpoint bytes. Training values repeat by scoring partition and must not be summed repeatedly. AE scoring includes conversion of reconstruction differences to float64 before averaging; training uses float32 MSE and AdamW with gradient clipping at 10.\n\n'
    summary+='## Interpretation limits\n\nThe four-epoch budget is deliberately bounded. Frozen preprocessing was originally fitted on all TRAIN classes, including known malicious traffic, as required by the unchanged baseline protocol; only anomaly-model fitting is benign-only. Thus this is a benign-trained anomaly branch on a shared frozen representation, not a wholly benign-fitted preprocessing pipeline. Benign labels are necessarily used to identify fitting and calibration cohorts, but neither binary nor family labels enter the model input or reconstruction objective.\n\n'
    summary+='Protocol A keeps Infilteration malicious scores/labels out of representation selection and threshold fitting. Protocol B explicitly uses validation Infilteration labels. Prior phases have already exposed this test cohort, so these development results are not a fresh confirmatory test or proof of general unseen-threat detection. The optional temporal autoencoder was deferred. No graph model, graph+temporal integration, explainability, risk fusion or full AIGT-CTD was introduced. Work stops after this seed-42 Phase 7.\n'
    (OUT/'phase7_results_summary.md').write_text(summary,encoding='utf-8')
    guard()
    checks=dict(frozen_data_hashes_verified=True,prior_phases_unchanged=True,full_benign_train_used_by_every_AE_epoch=True,
        representation_fit_train_only=True,preprocessing_unchanged=True,labels_excluded_from_inputs=True,
        architecture_and_early_stopping_benign_validation_only=True,protocol_A_benign_calibration_only=True,
        protocol_A_locked_before_malicious_scoring=True,protocol_B_explicitly_label_aware=True,test_excluded_from_selection=True,
        full_matched_test_cohort=True,seed_42_only=True)
    for name in PROTOCOL['architectures']:
        for epoch in read(METRICS/f'{name}_history.json'):
            require(epoch['training_rows']==training_manifest['benign_train_rows'],'AE subsampling detected')
    import torch,sklearn,pyarrow
    json_write(METRICS/'integrity_verification.json',dict(status='PASS',checks=checks,protocol_A_lock_sha256=sha256(LOCK_A),protocol_B_lock_sha256=sha256(LOCK_B),
        source_code_sha256={p.name:sha256(p) for p in (ROOT/'src/anomaly').glob('*.py')},
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__,pyarrow=pyarrow.__version__)))
    (OUT/'integrity_report.md').write_text('# Phase 7 integrity\n\n'+table([dict(check=k,status='PASS') for k in checks],['check','status'])+'\n\nEvidence: anomaly_training_manifest.json, configs/protocol.json, both calibration locks, complete epoch histories, per-model training/scoring manifests and metrics/integrity_verification.json. Original preprocessing was fitted on all TRAIN classes; it was not refitted or represented as benign-only.\n',encoding='utf-8')
    print('Phase 7 complete; stopping before any integration',flush=True)

if __name__=='__main__': report()
