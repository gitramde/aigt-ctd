"""Generate Phase 5 deliverables from locked, saved seed-42 outputs."""
import time
import numpy as np
from .data import *
from .evaluate import BASELINES, LOCK, baseline_scores
from src.phase4.metrics import binary_metrics


def markdown(rows, columns):
    def fmt(v):
        return f'{v:.6f}' if isinstance(v, float) else str(v)
    return '\n'.join(['| ' + ' | '.join(columns) + ' |',
                      '| ' + ' | '.join(['---'] * len(columns)) + ' |'] +
                     ['| ' + ' | '.join(fmt(r.get(c, '')) for c in columns) + ' |' for r in rows])


def report():
    print('Verifying frozen data and experiment provenance', flush=True)
    prepare()  # Includes full raw/cleaned/membership and Phase 4/4A hash checks.
    selected = read(OUT/'selected_model.json')
    lock = read(LOCK)
    manifest = read(METRICS/'sequence_manifest.json')
    require(sha256(OUT/'selected_model.json') == lock['selected_model_sha256'], 'Selection changed')
    require(sha256(METRICS/'sequence_manifest.json') == lock['cohort_manifest_sha256'], 'Cohorts changed')
    require(sha256(OUT/'thresholds.csv') == lock['threshold_csv_sha256'], 'Thresholds changed')
    names = [selected['config']['name'], 'transformer_L1_control']
    evaluations = {}
    runtime = []
    all_metrics = []
    families = []
    for name in names:
        record = read(METRICS/f'{name}_evaluation.json')
        training = read(METRICS/f'{name}_training.json')
        require(record['threshold_lock_sha256'] == sha256(LOCK), 'Evaluation lock mismatch')
        require(record['model_sha256'] == sha256(MODELS/f'{name}.pt') == lock['artifacts'][name]['model_sha256'], 'Model mismatch')
        require(training['validation_prediction_sha256'] == sha256(METRICS/f'{name}_validation.npy'), 'Selection predictions changed')
        evaluations[name] = record
        for part, entry in record['partitions'].items():
            require(sha256(entry['prediction_path']) == entry['prediction_sha256'], 'Predictions changed')
            all_metrics.extend(entry['metrics'])
            requested = ('Infilteration', 'Brute Force -Web', 'Brute Force -XSS', 'SQL Injection') if part == 'validation' else ('Bot', 'Infilteration')
            families.extend(r for r in entry['families'] if r['attack_family'] in requested)
            runtime.append(dict(model=name, partition=part, seed=42,
                training_seconds=training['training_seconds'], selection_seconds=training['selection_seconds'],
                training_stage_seconds=training['training_stage_seconds'],
                training_peak_memory_bytes=training['peak_memory_bytes'], evaluation_peak_memory_bytes=record['peak_memory_bytes'],
                model_size_bytes=training['model_size_bytes'], **entry['runtime']))
    for part in ('validation', 'test'):
        write(f'{part}_metrics.csv', [r for r in all_metrics if r['partition'] == part])
    write('attack_family_diagnostics.csv', families)
    write('runtime_metrics.csv', runtime)
    write('temporal_ablation.csv', all_metrics)

    comparison = []
    book = manifest['codebook']
    for part in ('validation', 'test'):
        c = cohort(part)
        for name in list(BASELINES) + names[:1]:
            p = baseline_scores(name, part) if name in BASELINES else np.load(evaluations[name]['partitions'][part]['prediction_path'])
            require(len(p) == len(c['target_binary']), 'Matched target count differs')
            rank = binary_metrics(c['target_binary'], p)
            for t in lock['thresholds']:
                if t['model'] != name or t['criterion'] not in ('fixed_0_5', 'fpr_at_most_0.01'):
                    continue
                m = binary_metrics(c['target_binary'], p, t['threshold'], include_auc=False)
                row = dict(model=name, partition=part, criterion=t['criterion'], threshold=t['threshold'],
                    cohort_sha256=manifest['partitions'][part]['cohort_sha256'], **{k:m[k] for k in ('records','recall','precision','f1','false_positive_rate')},
                    roc_auc=rank['roc_auc'], average_precision=rank['average_precision'])
                for family in ('Bot', 'Infilteration'):
                    mask = c['family_code'] == book.index(family)
                    row[family+'_N'] = int(mask.sum())
                    row[family+'_recall'] = float((p[mask] >= t['threshold']).mean()) if mask.any() else None
                comparison.append(row)
    write('matched_cohort_baseline_comparison.csv', comparison)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, criterion in zip(axes, ('fixed_0_5', 'fpr_at_most_0.01')):
        rows = [r for r in comparison if r['partition'] == 'test' and r['criterion'] == criterion]
        ax.barh([r['model'].replace('transformer_L64_d64_n2_p01', 'Temporal Transformer') for r in rows], [r['recall'] for r in rows])
        ax.set(xlim=(0,1), xlabel='Malicious recall on matched test targets', title=criterion)
    fig.savefig(FIGURES/'matched_test_recall.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for ax, row in zip(axes.flat, [r for r in all_metrics if r['criterion'] == 'fixed_0_5']):
        matrix = np.array([[row['true_negatives'],row['false_positives']], [row['false_negatives'],row['true_positives']]])
        ax.imshow(np.log1p(matrix), cmap='Blues')
        for (i,j), count in np.ndenumerate(matrix): ax.text(j,i,f'{count:,}',ha='center',va='center')
        ax.set(xticks=[0,1], yticks=[0,1], xlabel='Predicted (0 benign, 1 malicious)', ylabel='Actual', title=row['model']+'\n'+row['partition'])
    fig.savefig(FIGURES/'confusion_matrices.png', dpi=160); plt.close(fig)

    counts = [dict(partition=p, **{k:v[k] for k in ('rows','targets','stride','benign','malicious','coverage_fraction')}) for p,v in manifest['partitions'].items()]
    sequence_text = '''# Phase 5 sequence construction

Frozen membership is already chronological; its original timestamps were scanned for monotonic order and valid calendar days before training. Ties retain frozen membership order. Each sequence contains contiguous historical flow vectors and its final flow; only that final flow supplies the binary target (0 benign, 1 malicious) and diagnostic family. No sequence includes future records or crosses a partition/day boundary. Quarantined 1970 records remain excluded.

Lengths 16, 32 and 64 share target records: exclude the first 63 records of every day for all lengths, then take training stride 128 and validation/test stride 8. This common cohort makes length selection and L1 ablation comparable. Striding reduces CPU work and retains deterministic chronological coverage, but metrics describe sampled targets, not every frozen record. Training batches shuffle already-constructed windows, never their internal temporal order. The L1 control uses identical target records and parameter count, with only the final vector and the same final positional index.

'''+markdown(counts, ['partition','rows','targets','stride','benign','malicious','coverage_fraction'])+'\n'
    (OUT/'sequence_construction_report.md').write_text(sequence_text, encoding='utf-8')
    checks = dict(manifest['checks'], full_frozen_hash_verification=True, phase4_and_phase4a_unchanged=True,
        preprocessing_train_fitted_only=True, targets_and_families_excluded_from_features=True,
        timestamps_ordering_only=True, test_excluded_from_selection=not lock['selection_uses_test'],
        thresholds_and_selected_models_verified=True, matched_baseline_targets=True)
    excluded = {'target_binary','attack_label','timestamp','source_row','source_file','file_date'}
    require(not excluded.intersection(manifest['features']), 'Metadata in features')
    require(all(checks.values()), 'Leakage/integrity check failed')
    integrity()
    json_write(METRICS/'output_validation.json',dict(status='PASS',checked_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),checks=checks,
        threshold_lock_sha256=sha256(LOCK), baseline_snapshot_sha256=sha256(METRICS/'baseline_snapshot.json')))
    (OUT/'integrity_report.md').write_text('# Phase 5 integrity verification\n\n'+markdown([dict(check=k,status='PASS') for k in checks],['check','status'])+'''

Full verification rehashes original raw data, cleaned/quarantine artifacts and frozen membership through the Phase 4 guard. The pretraining integrity record and baseline snapshot preserve the earlier pretraining checks. The snapshot also protects Phase 4/4A results and source files. Evaluation records bind saved predictions to the model and operating-threshold lock. Partition chronology and day spans establish training/evaluation separation. The unchanged frozen preprocessing and feature list exclude labels, families and ordering metadata.

Thresholds were selected on matched validation predictions and frozen before held-out evaluation. Baseline models were not retrained; their existing saved prediction arrays were indexed by the exact same frozen partition row indices. Their 1% FPR thresholds are reselected on matched validation targets, not copied from a different cohort. A validation FPR cap is not a guarantee of the same test FPR.
''', encoding='utf-8')
    test_rows = [r for r in comparison if r['partition']=='test']
    ablation = [r for r in all_metrics if r['criterion']=='fixed_0_5']
    summary = '# Phase 5 seed-42 development results\n\n'
    summary += f"Selected length {selected['config']['length']}, d_model {selected['config']['d_model']}, {selected['config']['layers']} encoder layers, 4 attention heads, dropout {selected['config']['dropout']}; epoch {selected['selected_epoch']}, validation macro-F1 {selected['selection_score']:.6f}. Four temporal configurations were evaluated, followed by one capacity-matched Transformer-L1 control. All configurations used at most three epochs, patience two, AdamW and class weighting derived only from training targets.\n\n"
    summary += 'Validation malicious traffic is dominated by UNSEEN Infilteration. Selection therefore measures later-period temporal generalization, not ordinary IID classification. Test outcomes were not used for model, sequence-length, epoch or threshold selection.\n\n'
    summary += '## Matched-cohort test comparison\n\n'+markdown(test_rows,['model','criterion','recall','precision','f1','false_positive_rate','roc_auc','average_precision','Bot_recall','Infilteration_recall'])+'\n\n'
    summary += '## Transformer-L1 control at threshold 0.5\n\n'+markdown(ablation,['model','partition','recall','precision','f1','macro_f1','false_positive_rate','roc_auc','average_precision'])+'\n\n'
    temporal = next(r for r in ablation if r['model']==names[0] and r['partition']=='test')
    control = next(r for r in ablation if r['model']==names[1] and r['partition']=='test')
    summary += f"Temporal minus L1 test differences at 0.5: recall {temporal['recall']-control['recall']:+.6f}, macro-F1 {temporal['macro_f1']-control['macro_f1']:+.6f}, AP {temporal['average_precision']-control['average_precision']:+.6f}. These descriptive differences do not establish a reliable temporal-context benefit or statistical significance.\n\n"
    summary += '''## Interpretation and measurement limits

This bounded, CPU-only, single-seed development run uses sparse target coverage and a three-epoch budget. Findings are conditional on that budget and cohort; they do not establish an optimized Transformer result. Family diagnostics mark training presence as KNOWN/UNSEEN and leave absent-family recall blank. Unseen families are not characterized as zero-day attacks. All five operating points are in validation_metrics.csv and test_metrics.csv; PR-AUC is trapezoidal precision-recall area, while AP is reported separately. Confusion-matrix columns are TN, FP, FN and TP.

runtime_metrics.csv separates training, validation selection, feature loading and inference. Inference latency is milliseconds per 1,000 evaluated target flows, including historical-window processing; end-to-end timing additionally includes frozen feature transformation. Peak memory is process peak working set/RSS, not parameter memory, and includes loaded feature arrays. Serialized size is checkpoint bytes. Training measurements repeat across the two evaluation partitions and must not be summed twice.

All tables and figures are generated programmatically from saved experiment outputs. No Phase 4/4A artifacts were modified. No graph, GATv2, anomaly branch, explainability, risk fusion, full AIGT-CTD or final five-seed experiment was introduced.
'''
    (OUT/'phase5_results_summary.md').write_text(summary, encoding='utf-8')
    print('Phase 5 reports complete:', OUT, flush=True)


if __name__ == '__main__':
    report()
