"""Generate all published tables and figures from persisted experiment outputs."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import BASE,METRICS,FIGURES,MODEL_NAMES,guard
from .evaluate import freeze
from src.baseline.common import csv_write,json_write,sha256,require


def run():
    freeze();guard()
    rows={p:[] for p in ('validation','test')};families=[];runtime=[]
    cm_dir=METRICS/'confusion_matrices';cm_dir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    for column,name in enumerate(MODEL_NAMES):
        training=json.loads((METRICS/f'{name}_seed42_training.json').read_text())
        result=json.loads((METRICS/f'{name}_seed42_evaluation.json').read_text())
        require(result['model_sha256']==training['model_sha256'],'Mismatched model outputs')
        for index,part in enumerate(('validation','test')):
            output=result['partitions'][part];m=output['metrics']
            require(sum(m[k] for k in ('true_negatives','false_positives','false_negatives','true_positives'))==m['records'],'Confusion matrix total mismatch')
            require(sum(f['total_records'] for f in output['families'])==m['true_positives']+m['false_negatives'],'Family/target count mismatch')
            require(sum(f['detected_as_malicious'] for f in output['families'])==m['true_positives'],'Family detection count mismatch')
            if part=='validation':
                require(abs(m['macro_f1']-training['selected_validation_macro_f1'])<1e-12,'Selected checkpoint evaluation differs')
            require(sha256(output['probability_path'])==output['probability_sha256'],'Predictions changed')
            rows[part].append(dict(model=name,seed=42,**m))
            families.extend(dict(model=name,seed=42,partition=part,**family) for family in output['families'])
            runtime.append(dict(model=name,seed=42,partition=part,training_seconds=training['training_seconds'],
                training_stage_wall_seconds=training['training_stage_wall_seconds'],selection_seconds=training['selection_seconds'],
                selected_step=training['selected_step'],executed_steps=training['executed_steps'],
                training_peak_memory_bytes=training['peak_memory_bytes'],evaluation_peak_memory_bytes=result['evaluation_peak_memory_bytes'],
                peak_memory_bytes=max(training['peak_memory_bytes'],result['evaluation_peak_memory_bytes']),
                model_size_bytes=training['model_size_bytes'],**output['runtime']))
            matrix=np.array([[m['true_negatives'],m['false_positives']],[m['false_negatives'],m['true_positives']]])
            csv_write(cm_dir/f'{name}_seed42_{part}.csv',[
                dict(actual='benign',predicted_benign=int(matrix[0,0]),predicted_malicious=int(matrix[0,1])),
                dict(actual='malicious',predicted_benign=int(matrix[1,0]),predicted_malicious=int(matrix[1,1]))])
            ax=axes[index,column];ax.imshow(matrix,cmap='Blues')
            for i in range(2):
                for j in range(2): ax.text(j,i,f'{matrix[i,j]:,}',ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
            ax.set(xticks=[0,1],xticklabels=['Benign','Malicious'],yticks=[0,1],yticklabels=['Benign','Malicious'],
                   xlabel='Predicted',ylabel='Actual',title=f'{name.replace("_"," ")}\n{part}')
    fig.suptitle('Seed 42 development: binary confusion matrices');fig.savefig(FIGURES/'confusion_matrices_seed42.png',dpi=160);plt.close(fig)
    for part in rows: csv_write(METRICS/f'baseline_{part}_metrics.csv',rows[part])
    csv_write(METRICS/'attack_family_diagnostics.csv',families)
    csv_write(METRICS/'unseen_attack_diagnostics.csv',[r for r in families if r['training_status']=='UNSEEN'])
    csv_write(METRICS/'infilteration_bot_diagnostics.csv',[r for r in families if r['attack_family'] in ('Infilteration','Bot')])
    csv_write(METRICS/'runtime_metrics.csv',runtime)
    check=json.loads((METRICS/'pretraining_check.json').read_text())
    lines=['# Binary baseline development results','',
        'Seed 42 only. These are development results; final five-seed experiments have not been run. Mean and standard deviation are therefore not estimated.', '',
        'The frozen baseline_v1 data, chronological membership and fitted preprocessing are unchanged. The training cache reproduces the frozen float64 preprocessing checksum before casting model inputs to float32. Target: 0 benign, 1 malicious. Attack labels are used only for diagnostics.', '',
        '## Pre-training distribution','', '| Partition | Benign | Malicious | Total |','|---|---:|---:|---:|']
    counts=check['attack_family_counts']
    for part in ('train','validation','test'):
        dist=counts[part];n=sum(dist.values());b=dist.get('Benign',0)
        lines.append(f'| {part} | {b:,} | {n-b:,} | {n:,} |')
    lines.extend(['','| Original family | Train | Validation | Test |','|---|---:|---:|---:|'])
    for family in sorted(set().union(*(set(v) for v in counts.values()))):
        values=' | '.join(f'{counts[p].get(family,0):,}' for p in ('train','validation','test'))
        lines.append(f'| {family} | {values} |')
    lines.append('')
    for part in ('validation','test'):
        unseen=[family for family,n in counts[part].items() if n and not counts['train'].get(family,0)]
        lines.append(f'Unseen in {part}: '+', '.join(unseen)+'.')
    lines.extend(['','## Evaluation','',
        'All checkpoint selection uses validation macro-F1 with a fixed probability threshold of 0.5. All four selected models are checksum-locked before any test inference. Test scores do not select configurations. PR-AUC is trapezoidal precision-recall area; average precision is also reported separately. Undefined precision/F1 denominators are assigned zero.', '',
        '| Model | Partition | Accuracy | Precision | Recall | F1 | Macro-F1 | ROC-AUC | PR-AUC | FPR | FNR |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for part in rows:
        for r in rows[part]:
            values=' | '.join(f'{r[k]:.6f}' for k in ('accuracy','precision','recall','f1','macro_f1','roc_auc','pr_auc','false_positive_rate','false_negative_rate'))
            lines.append(f'| {r["model"]} | {part} | {values} |')
    for part in rows:
        rates=[r['malicious_recall'] for r in rows[part]]
        lines.extend(['',f'At the fixed threshold, {part} malicious recall ranges from {min(rates):.2%} to {max(rates):.2%}. Read accuracy together with malicious recall and false-negative counts; it does not summarize attack detection on its own.'])
    lines.extend(['','## Infilteration and Bot binary detection','',
        '| Model | Partition | Family | Status | Total | Detected | Missed | Detection recall |','|---|---|---|---|---:|---:|---:|---:|'])
    for r in families:
        if r['attack_family'] in ('Infilteration','Bot'):
            recall='absent' if r['binary_detection_recall'] is None else f'{r["binary_detection_recall"]:.6f}'
            lines.append(f'| {r["model"]} | {r["partition"]} | {r["attack_family"]} | {r["training_status"]} | {r["total_records"]:,} | {r["detected_as_malicious"]:,} | {r["missed"]:,} | {recall} |')
    lines.extend(['','These results measure binary detection. Successful detection of a family absent from training is neither multiclass classification nor proof of zero-day detection. Absent families have no recall estimate.', '',
        '## Resource measurements','',
        'runtime_metrics.csv reports fitting time separately from validation selection and total training-stage time. Inference includes batched loading and frozen preprocessing; prediction-only time is also supplied. Latency is amortized milliseconds per 1,000 records. Peak memory is worker RSS/Windows peak working set, including imported libraries and data; model size is the serialized artifact size. Training measurements repeated on validation/test rows refer to the same fit.', '',
        '## Reproducibility and development limits','',
        'Run `python -m src.phase4.run` from the repository using the documented workspace-local runtime. Completed model stages are reused only when their configuration and artifact hashes match. An interrupted model stage restarts that model; this is not a mid-epoch recovery mechanism.', '',
        'Logistic regression uses SGDClassifier with logistic loss, L2 regularization and incremental fitting. MLP uses weighted incremental Adam fitting. Every epoch visits every training row; optimizer shuffling stays inside the fixed training partition. Random Forest uses the full training pool with a capped bootstrap draw per tree. XGBoost uses a quantile matrix and CPU histogram trees. All models use training-derived class/sample weighting. No validation/test oversampling occurs.', '',
        'Exact hyperparameters and checkpoint budgets are in configs/phase4_seed42.json. Per-model selection histories, environment versions, executed and selected steps, timing and model hashes are persisted in metrics/. Configuration/model locks are in development_configs_frozen.json. These bounded development configurations are not an exhaustive search.', '',
        'Final seeds 42, 123, 456, 789 and 1024 remain deferred. After final configuration freeze, all final runs must be executed before reporting mean ± standard deviation. Do not select further configurations using these test results.', '',
        '![Confusion matrices](figures/confusion_matrices_seed42.png)',''])
    lines.extend(['## Measured runtime','',
        '| Model | Partition | Fit seconds | Inference seconds | ms / 1,000 | Peak GiB | Model MiB |',
        '|---|---|---:|---:|---:|---:|---:|'])
    for r in runtime:
        lines.append(f'| {r["model"]} | {r["partition"]} | {r["training_seconds"]:.2f} | {r["inference_seconds"]:.2f} | {r["inference_latency_ms_per_1000"]:.3f} | {r["peak_memory_bytes"]/2**30:.3f} | {r["model_size_bytes"]/2**20:.4f} |')
    lines.append('')
    (BASE/'baseline_results_summary.md').write_text('\n'.join(lines),encoding='utf-8')
    guard(full=True);json_write(METRICS/'phase4_integrity_check.json',dict(status='PASS',frozen_artifacts_unchanged=True,seed=42,models=list(MODEL_NAMES)))
    print('REPORT COMPLETE; frozen artifact integrity PASS',flush=True)

if __name__=='__main__': run()
