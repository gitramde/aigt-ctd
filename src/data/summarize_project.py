"""Generate the project README from completed phase reports (no experiments run)."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/baseline_v1'
DIAG=BASE/'diagnostics'
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def table(headers,values):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+['| '+' | '.join(map(str,r))+' |' for r in values]
def main():
    cleaning=[r for r in rows(BASE/'cleaning_summary.csv') if r['filename']!='ALL'];plan=read(BASE/'split_plan.json');prep=read(BASE/'preprocessing.json')
    validation=rows(BASE/'metrics/baseline_validation_metrics.csv');test=rows(BASE/'metrics/baseline_test_metrics.csv')
    train=rows(DIAG/'training_partition_metrics.csv');transfer=rows(DIAG/'threshold_transfer_to_test.csv')
    quality=rows(ROOT/'results/nan_analysis.csv');infinity=rows(ROOT/'results/infinity_analysis.csv')
    raw=sum(int(r['raw_rows']) for r in cleaning);headers=sum(int(r['repeated_headers']) for r in cleaning)
    duplicates=sum(int(r['exact_nonheader_duplicates']) for r in cleaning);temporal=sum(int(r['temporal_rows']) for r in cleaning)
    raw_duplicates=sum(int(r['extra_occurrences']) for r in rows(ROOT/'results/duplicate_analysis.csv') if r['scope']=='file_class')
    epoch_exclusions=sum(int(r['year_1970_temporal_exclusions']) for r in cleaning)
    assert temporal==sum(p['rows'] for p in plan['partitions'].values())
    assert raw-headers-duplicates-epoch_exclusions==temporal
    assert raw_duplicates>=duplicates
    nan=sum(int(r['count']) for r in quality if r['scope']=='dataset_column')
    inf=sum(int(r['count']) for r in infinity if r['scope']=='dataset_column')
    rf=next(r for r in transfer if r['model']=='random_forest' and r['criterion']=='fpr_at_most_0.01')
    names={'logistic_regression':'Logistic Regression','random_forest':'Random Forest','xgboost':'XGBoost','mlp':'MLP'}
    lines=['# AIGT-CTD','', '**AI/ML-Based Dynamic Graph-Transformer Framework for Explainable Threat Detection**','',
        'Research project using CSE-CIC-IDS2018. Completed work covers dataset auditing, frozen chronological baseline preparation, four seed-42 binary development baselines, and temporal-generalization diagnostics. The proposed graph/transformer framework has not been implemented.','',
        '## Progress by phase','']
    lines+=table(['Phase','Completed work','Detailed report'],[
        ['1 - Dataset audit','Inventory and full-record quality audit of ten raw CSV files','[Historical audit workflow](docs/phase1_audit_history.md)'],
        ['2 - Data quality and experiment design','NaN/infinity, duplicates, timestamp anomalies, class distribution and graph-window feasibility','[Phase 2 audit](results/phase2_audit_notes.md)'],
        ['3 - Frozen baseline preparation','Cleaning, chronological split, leakage exclusions and training-only preprocessing','[Preparation report](results/baseline_v1/baseline_preparation_report.md)'],
        ['4 - Binary development baselines','Logistic Regression, Random Forest, XGBoost and MLP; seed 42 only','[Baseline results](results/baseline_v1/baseline_results_summary.md)'],
        ['4A - Temporal generalization diagnosis','Existing-model inference, score/ranking/calibration analysis, validation-only threshold transfer and feature shift','[Phase 4A analysis](results/baseline_v1/diagnostics/phase4a_temporal_generalization_analysis.md)']])
    lines += ['', '**Current scope:** target `0 = benign`, `1 = malicious`. Original attack-family labels are retained for diagnostics. Final five-seed experiments, graph construction, LSTM, Transformer, GATv2, anomaly detection, SHAP and AIGT-CTD remain unimplemented or unexecuted.','',
        '## Phases 1-2: data quality and graph feasibility','',
        f'The ten original files contain **{raw:,} records**. Nine have 80 columns; the February 20 file has 84. Raw files remain unchanged. Phase 2 resolves the detailed audit accounting used by subsequent preparation; older Phase 1 report-schema caveats are preserved in the historical workflow.','']
    lines+=table(['Finding','Result'],[
        ['Explicit NaN cells',f'{nan:,}; Flow Byts/s'],['Infinite cells',f'{inf:,}; Flow Byts/s and Flow Pkts/s'],
        ['Repeated-header records',f'{headers:,}'],['Extra exact duplicates in the raw audit',f'{raw_duplicates:,}, including {raw_duplicates-duplicates:,} repeated-header duplicates'],
        ['Non-header duplicates removed after header removal',f'{duplicates:,}'],['Invalid year-1970 timestamps',f'{epoch_exclusions:,}; quarantined from temporal experiments']])
    lines += ['', 'The duplicate and repeated-header counts overlap: removing all headers first leaves the smaller non-header duplicate count. Source row order is not chronological; the audit found backward timestamp transitions. Recorded times lack timezone/AM-PM information, so chronology follows the supplied timestamps without reconstructing missing clock information.','',
        'Only `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` retains source IP, destination IP and source port. The other nine files cannot independently supply host-to-host interaction graphs. February 20 contains benign traffic and DDoS attacks-LOIC-HTTP only.','',
        'The Phase 2 audit recommends **1 minute** as the primary graph-window candidate, with **5 and 10 minutes** as comparisons. Repeated endpoint-pair flows occur in all three window sizes. These are dataset-based design candidates, not validated graph-model results. See the [February 20 audit](results/feb20_graph_audit.csv) and [temporal window analysis](results/temporal_window_analysis.csv).','',
        'Detailed quality outputs: [NaNs](results/nan_analysis.csv), [infinities](results/infinity_analysis.csv), [duplicates](results/duplicate_analysis.csv), [timestamp anomalies](results/timestamp_anomalies.csv), and [attack distribution](results/attack_distribution.csv).','',
        '## Phase 3: frozen baseline_v1','',
        f'After removing headers and within-file exact duplicates, **{sum(int(r["cleaned_rows"]) for r in cleaning):,} records** remain. Excluding the 14 quarantined epoch-date records leaves **{temporal:,} temporally eligible records**. Infinity values in the two flow-rate columns were converted to NaN. Every removal/exclusion is audited.','']
    partition_rows=[]
    for p in ('train','validation','test'):
        v=plan['partitions'][p];b=v['class_counts']['Benign'];partition_rows.append([p,', '.join(v['dates']),f'{b:,}',f'{v["rows"]-b:,}',f'{v["rows"]:,}'])
    lines+=table(['Partition','Dates in 2018','Benign','Malicious','Total'],partition_rows)
    lines += ['', '**All malicious test records are from attack families absent from training:** Bot (282,310) and Infilteration (93,040). Validation malicious traffic is dominated by unseen Infilteration (68,857); its known-family examples are Brute Force -Web (362), Brute Force -XSS (151), and SQL Injection (53). Known-family test recall is therefore undefined. The dataset spelling `Infilteration` is preserved.','',
        f'The frozen preprocessor produces **{len(prep["output_feature_names"])} inputs**: {len(prep["numeric_columns"])} numeric features and three Protocol one-hot columns. Numeric mean imputation, categorical mode/vocabulary and scaling were fitted on training data only, then applied unchanged to validation/test. Labels, target fields, timestamps and excluded identifiers are not model inputs; destination port remains an included feature.','',
        'Exact membership rules and hashes are in [split_plan.json](results/baseline_v1/split_plan.json); fitted parameters are in [preprocessing.json](results/baseline_v1/preprocessing.json). The [cleaning audit](results/baseline_v1/cleaning_audit.csv), [feature exclusions](results/baseline_v1/feature_exclusion_audit.csv), and [verification](results/baseline_v1/verification.json) record preparation checks.','',
        '## Phase 4: seed-42 development results','',
        'Models were fitted only on training rows and selected using validation macro-F1. Selected model/configuration hashes were locked before test inference. Class weighting was used; validation/test were not oversampled. The fixed classification threshold was **0.5**.','',
        'Logistic Regression uses incremental SGD optimization of logistic loss; MLP uses incremental weighted Adam. Random Forest uses the full training pool with a capped bootstrap draw per tree. XGBoost uses batched quantile-matrix construction and CPU histogram trees. A memory-mapped float32 training cache reproduces the frozen float64 preprocessing checksum before conversion. These are bounded development runs, not exhaustive model searches.','']
    lines+=table(['Model','Validation macro-F1','Test accuracy','Test precision','Test recall','Test F1','Test ROC-AUC','Test AP'],[
        [names[r['model']],f'{float(next(v for v in validation if v["model"]==r["model"])["macro_f1"]):.4f}']+
        [f'{float(r[k]):.4f}' for k in ('accuracy','precision','recall','f1','roc_auc','average_precision')] for r in test])
    lines += ['', f'At threshold 0.5, test malicious recall spans **{min(float(r["recall"]) for r in test):.2%}-{max(float(r["recall"]) for r in test):.2%}**, and every model misses all Bot records. Overall accuracy conceals this poor detection. PR-AUC and average precision (AP) are distinct measures; the full reports retain both.','',
        'Saved outputs: [models](results/baseline_v1/models/), [validation metrics](results/baseline_v1/metrics/baseline_validation_metrics.csv), [test metrics](results/baseline_v1/metrics/baseline_test_metrics.csv), [family diagnostics](results/baseline_v1/metrics/attack_family_diagnostics.csv), [runtime metrics](results/baseline_v1/metrics/runtime_metrics.csv), and [confusion matrices](results/baseline_v1/figures/confusion_matrices_seed42.png).','',
        '## Phase 4A: what explains the performance gap?','',
        '**No models were retrained or recalibrated.** Existing predictions were reused for validation/test; fitted models were evaluated on training rows. The frozen split and preprocessing were preserved.','']
    lines+=table(['Model','Training ROC-AUC','Training malicious recall','Training F1'],[[names[r['model']]]+[f'{float(r[k]):.6f}' for k in ('roc_auc','malicious_recall','f1')] for r in train])
    lines += ['', f'- **Training versus generalization:** aggregate training recall is {min(float(r["malicious_recall"]) for r in train):.2%}-{max(float(r["malicious_recall"]) for r in train):.2%}. This rules out a broad failure to fit the training distribution, but does not establish learning of every rare family or rule out overfitting.',
        '- **Known-family validation:** performance is variable and often poor even for families present during training. These groups contain only 362, 151 and 53 records, so the estimates have limited support.',
        '- **Ranking versus threshold:** Random Forest and MLP retain test ranking information despite near-zero recall at 0.5. Bot and Infilteration behave differently; pooled test metrics depend on family mixture.',
        f'- **Validation-only threshold transfer:** Random Forest\'s threshold selected under validation FPR <= 1% is {float(rf["threshold"]):.6f}. Applied unchanged to test, it yields **{float(rf["recall"]):.2%} recall, {float(rf["precision"]):.2%} precision and {float(rf["false_positive_rate"]):.2%} FPR**. Bot recall is {float(rf["bot_recall"]):.2%}, versus {float(rf["infilteration_recall"]):.2%} for Infilteration. Recovery is partial.',
        '- **Calibration:** all four models have negative validation Brier skill against the constant-prevalence reference, with reliability gaps. Brier score reflects discrimination as well as calibration; no calibrator was fitted.',
        '- **Feature shift:** exact full-record comparisons cover all 80 inputs, including comparisons against training benign and malicious groups. Bot differs strongly on destination port and timing/rate features. Infilteration also shifts between validation and test.',
        '- **Interpretation:** the evidence supports a combination of temporal/family generalization problems and threshold/calibration mismatch. Marginal shifts do not prove causality, and the study does not independently isolate these causes.','',
        'Five threshold objectives were selected exclusively on validation: maximum binary F1, maximum macro-F1, and FPR caps of 1%, 0.5%, and 0.1%. They were persisted before transfer to test. Test threshold curves are **post-hoc descriptive only**; no test result selected a model or threshold. Because Phase 4 test results had already been seen, Phase 4A is not a new untouched confirmatory evaluation.','',
        'Successful binary detection of a family absent from training is neither multiclass recognition nor proof of zero-day detection.','',
        'Outputs include all requested CSVs and 24 figures under [diagnostics/](results/baseline_v1/diagnostics/). See [training performance](results/baseline_v1/diagnostics/training_partition_metrics.csv), [locked validation thresholds](results/baseline_v1/diagnostics/validation_selected_thresholds.csv), [test transfer](results/baseline_v1/diagnostics/threshold_transfer_to_test.csv), [feature shift](results/baseline_v1/diagnostics/feature_distribution_shift.csv), [calibration](results/baseline_v1/diagnostics/calibration_metrics.csv), and [integrity verification](results/baseline_v1/diagnostics/integrity_verification.json).','',
        '## Reproduction and resuming','',
        'Run commands from the repository root. The audit workflow and the modeling runtime have different dependencies; do not assume the old `.venv` contains the model packages. Phase 4 used system Python 3.13.5 plus workspace-local scikit-learn 1.7.2 and XGBoost 3.0.5. Exact versions are recorded in [environment.json](results/baseline_v1/metrics/environment.json) and [requirements-phase4.txt](requirements-phase4.txt).','',
        '| Workflow | Instructions |','|---|---|',
        '| Initial audit | [Historical workflow](docs/phase1_audit_history.md) |',
        '| Phase 2 audit | [Verification and reproduction](results/phase2_audit_notes.md#verification-and-reproduction) |',
        '| Frozen preparation | [BASELINE_PIPELINE.md](docs/BASELINE_PIPELINE.md) |',
        '| Baseline development | [Seed-42 instructions](docs/phase4_seed42.md) |',
        '| Diagnostics only | [Phase 4A instructions](docs/phase4a_diagnostics.md) |','',
        'To regenerate diagnostics from existing artifacts without training:','',
        '```powershell','python -m src.phase4a.run','```','',
        'Completed training-inference and feature checkpoints can be reused. The earlier baseline command `python -m src.phase4.run` is a training workflow: it reuses verified completed model stages, but can train missing stages. An interrupted model fitting stage restarts that model; it does not resume mid-epoch. Do not rerun preparation into the frozen version.','',
        'Raw data, derived caches and Python environments remain local under ignored directories. A fresh clone requires the dataset and reconstructed/verified artifacts before diagnostics can run.','',
        '## Deferred work','',
        'Final experiments with seeds **42, 123, 456, 789 and 1024** have not run. No final mean +/- standard deviation is reported. A final experiment protocol must be explicitly frozen before those runs, and already-observed test outcomes must not drive further model or threshold selection. Graph construction and advanced models remain future work; their effectiveness has not been demonstrated.','',
        'This README summarizes saved results. Regenerate its tables with `python src/data/summarize_project.py`; that command does not run experiments or modify frozen artifacts.','']
    (ROOT/'README.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__': main()
