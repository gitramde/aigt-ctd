"""Readable summary of already verified Phase 11A results; no fitting."""
import pandas as pd
from .common import *

def table(frame):
    def fmt(x):
        if pd.isna(x):
            return 'null'
        if isinstance(x, float):
            return f'{x:.6e}' if 0 < abs(x) < 1e-6 else f'{x:.6f}'
        return str(x)
    columns = list(frame.columns)
    return '\n'.join(['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |'] +
                     ['| ' + ' | '.join(fmt(x) for x in row) + ' |' for row in frame.itertuples(index=False, name=None)])

def main():
    check_lock()
    verified=read(OUT/'audit/phase11a/final_verification.json')
    require(verified['status']=='PASS' and verified['completed_required_fits']==35,'Phase 11A verification incomplete')
    dest=OUT/'aggregate_phase11a'
    means=pd.read_csv(dest/'five_seed_mean_sample_sd.csv')
    families=pd.read_csv(dest/'family_mean_sample_sd.csv')
    runtime=pd.read_csv(dest/'runtime.csv')
    fits=runtime.drop_duplicates(['seed','model'])
    test=means[means.partition=='test']
    primary=test[test.criterion.isin(['fixed_0_5','fpr_at_most_0.01','benign_fpr_at_most_0.01','component_1_percent'])]
    compact=primary[['model','cohort','criterion','completed_seeds']].copy()
    for metric in ('recall','precision','f1','macro_f1','false_positive_rate','roc_auc','average_precision'):
        compact[metric+' mean ± SD']=[f'{a:.6f} ± {b:.6f}' for a,b in zip(primary[metric+'_mean'],primary[metric+'_std'])]
    fam=families[(families.partition=='test') & families.attack_family.isin(['Bot','Infilteration']) & families.criterion.isin(['fpr_at_most_0.01','benign_fpr_at_most_0.01','component_1_percent'])]
    costs=fits.groupby('model').agg(completed_seeds=('seed','nunique'),mean_fit_seconds=('training_seconds','mean'),mean_validation_selection_seconds=('validation_selection_seconds','mean'),maximum_stage_peak_bytes=('peak_memory_bytes','max'),mean_checkpoint_bytes=('model_size_bytes','mean')).reset_index()
    lines=['# Phase 11A execution summary', '', '**COMPLETE: 35/35 mandatory fits and all five same-seed RF/AE/OR comparisons.**', '',
           'Seeds: 42, 123, 456, 789, 1024. Models: logistic regression, random forest, XGBoost, MLP, Transformer-L1, Transformer-L64, and AE-B. The 20 completed final supervised fits were verified and retained. Fifteen temporal/anomaly fits were completed in this invocation. Seed 42 uses fresh final fits; no development model was substituted.', '',
           'No graph model was executed and Phase 11B was not started. The authoritative final_spec_v1 and protected development artifacts passed hash verification without modification.', '',
           '## Integrity and interruptions', '',
           'All 592 frozen artifact hashes, specification hashes, exact environment/hardware, runner/configuration locks, seed propagation, threshold semantics, cohort membership/order, causality, masks, and history identities passed before fitting. Saved checkpoint, score, ID and threshold-lock hashes passed verification. Metrics and family counts were recomputed from saved scores; confusion matrices and classification metrics were checked independently. Numerical thresholds were independently recalibrated under the frozen algorithms for verification.', '',
           'An earlier seed-42 Transformer-L64 attempt had stopped after an epoch without a completion record or resumable optimizer/RNG state. Its entire output directory and hashes are preserved under `audit/phase11a/interrupted_seed_42_temporal/`. The identical seed/configuration was rerun from initialization. The interruption is not a completed fit and its unknown total resource cost is not silently included in completed-run timing. An earlier pre-training seed-provenance failure remains preserved under `audit/attempt_01_stopped/`. No seeds were substituted or results omitted.', '',
           'No required Phase 11A run remains failed or incomplete. Historical interrupted/failed attempts remain part of the audit record.', '',
           '## Test results', '',
           'Values are arithmetic means ± sample SD (ddof=1), computed per seed before aggregation. Full-precision per-seed metrics and integer confusion counts are in `aggregate_phase11a/per_seed_metrics.csv`; do not pool repeated targets across seeds.', '', table(compact), '',
           'Matched conventional controls use the same-seed full-data predictions sliced to the frozen Phase5 targets with separately calibrated matched-validation thresholds. Comparisons across different cohorts are not component ablations. OR ranking metrics use its binary decision and are not equivalent to continuous RF/AE ranking performance.', '',
           '## Bot and Infilteration', '', table(fam[['model','cohort','criterion','attack_family','mean','std','count']]), '',
           'Bot and Infilteration are absent from TRAIN, but Infilteration occurs in validation, making supervised checkpoint/threshold selection label-aware for that family. These are attack families absent from training, not a zero-day-detection claim. Complete family counts/detections/recalls are in `per_seed_family_metrics.csv` and `bot_infilteration_diagnostics.csv`.', '',
           '## Temporal paired differences and RF/AE/OR', '',
           'Within-seed L64 minus L1 differences at every shared operating point, with mean and sample SD, are in `paired_per_seed_differences.csv` and `paired_mean_sample_sd.csv`. Positive FPR differences mean more false alarms; a favorable operating point alone does not establish consistent temporal benefit.', '',
           'The exact component-1% comparison, including actual test FPR and Bot/Infilteration recall, is in `rf_ae_or_comparison.csv`. RF-only, AE-only, both, neither and Jaccard are reported by family in `rf_ae_overlap.csv`, including benign overlap. An empty union has null Jaccard. Component validation caps do not guarantee a 1% OR union FPR. No fusion parameter was fitted.', '',
           '## Resource measurements', '', table(costs), '',
           'CPU-only Intel i7-8565U, 16,944,599,040 RAM bytes, two math/PyTorch threads, RF n_jobs=1, one fitting process at a time. `runtime.csv` retains per-seed, per-partition inference stage measurements, validation-selection and fit times, peak process memory, and checkpoint sizes. Peak memory is the maximum of separately measured fit/evaluation stage peaks, not their sum. Repeated training costs across validation/test rows are not independent costs.', '',
           'Timing comparisons must retain their cohort and stage scope. Model loading is separate; cached inference is not raw-data end-to-end processing. Frozen cleaning/cache construction was reused and not remeasured as new fitting. The full-data conventional, sampled temporal, and AE inference pipelines have different preparation costs.', '',
           '## Statistical scope', '',
           'The test cohorts were previously observed. Five seeds measure training stochasticity on fixed records, not independent datasets or split uncertainty. No significance tests, confidence intervals, new search, or confirmatory claims were introduced. Contrary evidence must remain reportable.', '',
           '**STOP after Phase 11A.**']
    (OUT/'PHASE_11A_EXECUTION_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
