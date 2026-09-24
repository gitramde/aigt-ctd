# AIGT-CTD

**AI/ML-Based Dynamic Graph-Transformer Framework for Explainable Threat Detection**

Research project using CSE-CIC-IDS2018 to evaluate flow-based, temporal, graph and benign-trained anomaly detection under explicitly separated experimental cohorts. Development phases 1-9, the Phase 10 experiment freeze, and five-seed final execution in Phases 11A and 11B are complete. Graph-temporal modeling uses a frozen GATv2 encoder followed by a separately trained Transformer head; explainability remains deferred.

## Progress by phase

| Phase | Completed work | Detailed report |
|---|---|---|
| 1 - Dataset audit | Inventory and full-record quality audit of ten raw CSV files | [Historical audit workflow](docs/phase1_audit_history.md) |
| 2 - Data quality and experiment design | NaN/infinity, duplicates, timestamp anomalies, class distribution and graph-window feasibility | [Phase 2 audit](results/phase2_audit_notes.md) |
| 3 - Frozen baseline preparation | Cleaning, chronological split, leakage exclusions and training-only preprocessing | [Preparation report](results/baseline_v1/baseline_preparation_report.md) |
| 4 - Binary development baselines | Logistic Regression, Random Forest, XGBoost and MLP; seed 42 only | [Baseline results](results/baseline_v1/baseline_results_summary.md) |
| 4A - Temporal generalization diagnosis | Existing-model inference, score/ranking/calibration analysis, validation-only threshold transfer and feature shift | [Phase 4A analysis](results/baseline_v1/diagnostics/phase4a_temporal_generalization_analysis.md) |
| 5 - Temporal development | Flow-sequence Transformer and capacity-matched L1 control; matched baseline comparisons | [Phase 5 results](results/temporal_v1/phase5_results_summary.md) |
| 6 - Graph contribution | February 20 GATv2, edge MLP, self-only and conventional controls; window sensitivities | [Phase 6 results](results/graph_v1/phase6_results_summary.md) |
| 7 - Anomaly branch | Benign-trained autoencoders and Isolation Forest; separate benign-only and label-aware calibration | [Phase 7 results](results/anomaly_v1/phase7_results_summary.md) |
| 8 - Frozen-score fusion | RF/AE MAX, weighted and OR comparisons without model refitting | [Phase 8 results](results/fusion_v1/phase8_results_summary.md) |
| 9 - Graph-temporal contribution | Frozen GATv2 representations with matched L1/L4/L8 Transformer heads | [Phase 9 results](results/graph_temporal_v1/phase9_results_summary.md) |
| 10 - Final specification freeze | Fixed model roster, cohorts, selection/calibration rules, metrics and claims; no training | [Final experiment specification](results/final_spec_v1/FINAL_EXPERIMENT_SPEC.md) |
| 11A - Final full-data and temporal runs | 35 fresh fits across five seeds; RF/AE OR derived from same-seed scores | [Phase 11A completion](results/final_runs_v1/PHASE_11A_EXECUTION_SUMMARY.md) |
| 11B - Final graph and graph-temporal runs | 20 mandatory plus 5 self-only fresh fits across five seeds | [Phase 11B completion and results](results/final_runs_v1/PHASE_11B_EXECUTION_SUMMARY.md) |

**Current scope:** target `0 = benign`, `1 = malicious`; original attack-family labels are retained for diagnostics. Final seeds are **42, 123, 456, 789 and 1024**, including freshly fitted seed 42. The completed approach tests component contributions and detection tradeoffs; it does not establish a unified end-to-end explainable detector.

Historical reports retain their phase-local stopping statements. In particular, `FINAL_EXECUTION_SUMMARY.md` records an earlier failed preflight, and the Phase 11A summary predates graph execution. The separate Phase 11A and Phase 11B completion reports linked above describe the completed final runs.

## Implemented approach

The study separates chronological generalization from within-day graph contribution because only February 20 retains the endpoint identifiers needed for host graphs. Metrics from these cohorts are not interchangeable component ablations.

| Cohort | Train / validation / test targets | Purpose |
|---|---|---|
| Full dataset | 12,795,136 / 1,652,943 / 1,374,143 | Flow baselines, anomaly detection and RF/AE fusion; later-period Bot/Infilteration diagnostics |
| Phase 5 matched | 99,962 / 206,603 / 171,753 | Flow-sequence Transformer versus L1 and baselines evaluated on identical targets |
| February 20 matched | 172,984 / 88,782 / 212,591 | GATv2 and graph-temporal contribution versus matched controls; Benign versus LOIC-HTTP only |

1. **Freeze data and preprocessing.** Clean and audit records, preserve chronological membership, exclude leakage-prone identifiers, and fit preprocessing on TRAIN only. The full-data and within-day graph experiments use their own training partitions and preprocessing.
2. **Measure flow and temporal baselines.** Fit Logistic Regression, Random Forest, XGBoost and MLP. Compare Transformer-L64 against an equal-capacity L1 control on identical targets; slice baseline predictions to that cohort and recalibrate thresholds on matched validation rows. Flow sequences follow start-time order and remain within each day/partition; they are not completion-causal online replay.
3. **Isolate graph contribution.** Split February 20 at 10:30 and 11:00. Use one-minute graph windows as primary, with five-/ten-minute development sensitivities. Targets start in the final 30 seconds of a minute; historical context includes only flows completed strictly before the second-30 cutoff and stays within its window and partition. Edge-aware GATv2 is compared with an edge MLP, a self-only GATv2 and conventional controls. IP identities define topology, not numeric model features.
4. **Test graph-derived temporal context.** Freeze the selected GATv2 encoder, concatenate source/destination embeddings with target flow features, then train Transformer heads. L1 is the capacity control; L8 includes the current flow plus seven preceding minute slots. Historical tokens prefer the same directed pair, then role-consistent source/destination fallback; missing minutes are masked. Each final head uses its same-seed encoder. The Phase 9 coverage-gate amendment remains a documented development limitation.
5. **Measure anomaly complementarity and fusion.** Fit the autoencoder on benign TRAIN records using the existing all-TRAIN preprocessor. Separate benign-validation FPR calibration from label-aware validation objectives. Isolation Forest and MAX/weighted fusion remain development comparisons; final fusion is same-seed RF OR AE. The OR union does not inherit its components' FPR cap, and its binary ranking metrics are distinct from continuous-score ranking metrics.
6. **Freeze and repeat.** Phase 10 fixes architectures, cohorts, budgets and calibration algorithms. Final runs repeat fresh fitting and validation-only checkpoint/threshold selection for each seed, lock thresholds before test evaluation, and report per-seed metrics, arithmetic means, sample SD (`ddof=1`) and paired component differences.

All test cohorts were observed during development. Five seeds measure training stochasticity on fixed splits, not split uncertainty, independent dataset replications or a new confirmatory holdout. Bot and Infilteration are absent from full-data TRAIN, but Infilteration labels participate in supervised validation selection. February 20 results address within-day LOIC-HTTP detection. Neither experiment establishes zero-day detection, multiclass recognition or strict online deployment performance.


## Phases 1-2: data quality and graph feasibility

The ten original files contain **16,233,002 records**. Nine have 80 columns; the February 20 file has 84. Raw files remain unchanged. Phase 2 resolves the detailed audit accounting used by subsequent preparation; older Phase 1 report-schema caveats are preserved in the historical workflow.

| Finding | Result |
|---|---|
| Explicit NaN cells | 59,721; Flow Byts/s |
| Infinite cells | 131,799; Flow Byts/s and Flow Pkts/s |
| Repeated-header records | 59 |
| Extra exact duplicates in the raw audit | 410,763, including 56 repeated-header duplicates |
| Non-header duplicates removed after header removal | 410,707 |
| Invalid year-1970 timestamps | 14; quarantined from temporal experiments |

The duplicate and repeated-header counts overlap: removing all headers first leaves the smaller non-header duplicate count. Source row order is not chronological; the audit found backward timestamp transitions. Recorded times lack timezone/AM-PM information, so chronology follows the supplied timestamps without reconstructing missing clock information.

Only `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` retains source IP, destination IP and source port. The other nine files cannot independently supply host-to-host interaction graphs. February 20 contains benign traffic and DDoS attacks-LOIC-HTTP only.

The Phase 2 audit recommends **1 minute** as the primary graph-window candidate, with **5 and 10 minutes** as comparisons. Repeated endpoint-pair flows occur in all three window sizes. These were audit-stage design candidates; Phase 6 subsequently evaluated all three with one minute predeclared primary. See the [February 20 audit](results/feb20_graph_audit.csv) and [temporal window analysis](results/temporal_window_analysis.csv).

Detailed quality outputs: [NaNs](results/nan_analysis.csv), [infinities](results/infinity_analysis.csv), [duplicates](results/duplicate_analysis.csv), [timestamp anomalies](results/timestamp_anomalies.csv), and [attack distribution](results/attack_distribution.csv).

## Phase 3: frozen baseline_v1

After removing headers and within-file exact duplicates, **15,822,236 records** remain. Excluding the 14 quarantined epoch-date records leaves **15,822,222 temporally eligible records**. Infinity values in the two flow-rate columns were converted to NaN. Every removal/exclusion is audited.

| Partition | Dates in 2018 | Benign | Malicious | Total |
|---|---|---|---|---|
| train | 2018-02-14, 2018-02-15, 2018-02-16, 2018-02-20, 2018-02-21, 2018-02-22 | 10,885,643 | 1,909,493 | 12,795,136 |
| validation | 2018-02-23, 2018-02-28 | 1,583,520 | 69,423 | 1,652,943 |
| test | 2018-03-01, 2018-03-02 | 998,793 | 375,350 | 1,374,143 |

**All malicious test records are from attack families absent from training:** Bot (282,310) and Infilteration (93,040). Validation malicious traffic is dominated by unseen Infilteration (68,857); its known-family examples are Brute Force -Web (362), Brute Force -XSS (151), and SQL Injection (53). Known-family test recall is therefore undefined. The dataset spelling `Infilteration` is preserved.

The frozen preprocessor produces **80 inputs**: 77 numeric features and three Protocol one-hot columns. Numeric mean imputation, categorical mode/vocabulary and scaling were fitted on training data only, then applied unchanged to validation/test. Labels, target fields, timestamps and excluded identifiers are not model inputs; destination port remains an included feature.

Exact membership rules and hashes are in [split_plan.json](results/baseline_v1/split_plan.json); fitted parameters are in [preprocessing.json](results/baseline_v1/preprocessing.json). The [cleaning audit](results/baseline_v1/cleaning_audit.csv), [feature exclusions](results/baseline_v1/feature_exclusion_audit.csv), and [verification](results/baseline_v1/verification.json) record preparation checks.

## Phase 4: seed-42 development results

Models were fitted only on training rows and selected using validation macro-F1. Selected model/configuration hashes were locked before test inference. Class weighting was used; validation/test were not oversampled. The fixed classification threshold was **0.5**.

Logistic Regression uses incremental SGD optimization of logistic loss; MLP uses incremental weighted Adam. Random Forest uses the full training pool with a capped bootstrap draw per tree. XGBoost uses batched quantile-matrix construction and CPU histogram trees. A memory-mapped float32 training cache reproduces the frozen float64 preprocessing checksum before conversion. These are bounded development runs, not exhaustive model searches.

| Model | Validation macro-F1 | Test accuracy | Test precision | Test recall | Test F1 | Test ROC-AUC | Test AP |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.4983 | 0.6924 | 0.0569 | 0.0081 | 0.0142 | 0.5514 | 0.3452 |
| Random Forest | 0.4907 | 0.7247 | 0.0687 | 0.0006 | 0.0012 | 0.8329 | 0.6772 |
| XGBoost | 0.4998 | 0.7249 | 0.2348 | 0.0032 | 0.0063 | 0.5539 | 0.5316 |
| MLP | 0.4947 | 0.7264 | 0.3002 | 0.0013 | 0.0027 | 0.7750 | 0.5127 |

At threshold 0.5, test malicious recall spans **0.06%-0.81%**, and every model misses all Bot records. Overall accuracy conceals this poor detection. PR-AUC and average precision (AP) are distinct measures; the full reports retain both.

Saved outputs: [models](results/baseline_v1/models/), [validation metrics](results/baseline_v1/metrics/baseline_validation_metrics.csv), [test metrics](results/baseline_v1/metrics/baseline_test_metrics.csv), [family diagnostics](results/baseline_v1/metrics/attack_family_diagnostics.csv), [runtime metrics](results/baseline_v1/metrics/runtime_metrics.csv), and [confusion matrices](results/baseline_v1/figures/confusion_matrices_seed42.png).

## Phase 4A: what explains the performance gap?

**No models were retrained or recalibrated.** Existing predictions were reused for validation/test; fitted models were evaluated on training rows. The frozen split and preprocessing were preserved.

| Model | Training ROC-AUC | Training malicious recall | Training F1 |
|---|---|---|---|
| Logistic Regression | 0.995084 | 0.999078 | 0.877324 |
| Random Forest | 0.999990 | 0.999369 | 0.989427 |
| XGBoost | 0.999992 | 0.999820 | 0.990713 |
| MLP | 0.999986 | 0.998618 | 0.996464 |

- **Training versus generalization:** aggregate training recall is 99.86%-99.98%. This rules out a broad failure to fit the training distribution, but does not establish learning of every rare family or rule out overfitting.
- **Known-family validation:** performance is variable and often poor even for families present during training. These groups contain only 362, 151 and 53 records, so the estimates have limited support.
- **Ranking versus threshold:** Random Forest and MLP retain test ranking information despite near-zero recall at 0.5. Bot and Infilteration behave differently; pooled test metrics depend on family mixture.
- **Validation-only threshold transfer:** Random Forest's threshold selected under validation FPR <= 1% is 0.173709. Applied unchanged to test, it yields **34.27% recall, 93.94% precision and 0.83% FPR**. Bot recall is 44.96%, versus 1.85% for Infilteration. Recovery is partial.
- **Calibration:** all four models have negative validation Brier skill against the constant-prevalence reference, with reliability gaps. Brier score reflects discrimination as well as calibration; no calibrator was fitted.
- **Feature shift:** exact full-record comparisons cover all 80 inputs, including comparisons against training benign and malicious groups. Bot differs strongly on destination port and timing/rate features. Infilteration also shifts between validation and test.
- **Interpretation:** the evidence supports a combination of temporal/family generalization problems and threshold/calibration mismatch. Marginal shifts do not prove causality, and the study does not independently isolate these causes.

Five threshold objectives were selected exclusively on validation: maximum binary F1, maximum macro-F1, and FPR caps of 1%, 0.5%, and 0.1%. They were persisted before transfer to test. Test threshold curves are **post-hoc descriptive only**; no test result selected a model or threshold. Because Phase 4 test results had already been seen, Phase 4A is not a new untouched confirmatory evaluation.

Successful binary detection of a family absent from training is neither multiclass recognition nor proof of zero-day detection.

Outputs include all requested CSVs and 24 figures under [diagnostics/](results/baseline_v1/diagnostics/). See [training performance](results/baseline_v1/diagnostics/training_partition_metrics.csv), [locked validation thresholds](results/baseline_v1/diagnostics/validation_selected_thresholds.csv), [test transfer](results/baseline_v1/diagnostics/threshold_transfer_to_test.csv), [feature shift](results/baseline_v1/diagnostics/feature_distribution_shift.csv), [calibration](results/baseline_v1/diagnostics/calibration_metrics.csv), and [integrity verification](results/baseline_v1/diagnostics/integrity_verification.json).

## Final five-seed outputs

Phase 11A completed four flow baselines, Transformer-L1/L64 and autoencoder-B for each seed (35 fits). Same-seed RF/AE scores also produce OR decisions without additional fitting. Phase 11B completed edge MLP, GATv2 and graph-Transformer-L1/L8 (20 fits), plus all five resource-conditional self-only GATv2 fits. L4, alternative graph windows, Isolation Forest retraining and MAX/weighted fusion were excluded from final fitting.

| Output | Phase 11A | Phase 11B |
|---|---|---|
| Per-seed metrics | [CSV](results/final_runs_v1/aggregate_phase11a/per_seed_metrics.csv) | [CSV](results/final_runs_v1/aggregate_phase11b/per_seed_metrics.csv) |
| Five-seed mean and sample SD | [CSV](results/final_runs_v1/aggregate_phase11a/five_seed_mean_sample_sd.csv) | [CSV](results/final_runs_v1/aggregate_phase11b/aggregate_metrics.csv) |
| Paired contribution differences | [CSV](results/final_runs_v1/aggregate_phase11a/paired_mean_sample_sd.csv) | [CSV](results/final_runs_v1/aggregate_phase11b/paired_aggregate_differences.csv) |
| Family metrics | [CSV](results/final_runs_v1/aggregate_phase11a/per_seed_family_metrics.csv) | [CSV](results/final_runs_v1/aggregate_phase11b/per_seed_family_metrics.csv) |
| Runtime measurements | [CSV](results/final_runs_v1/aggregate_phase11a/runtime.csv) | [CSV](results/final_runs_v1/aggregate_phase11b/runtime_metrics.csv) |

For the full-data family analysis, start with [Bot/Infilteration diagnostics](results/final_runs_v1/aggregate_phase11a/bot_infilteration_diagnostics.csv), [RF/AE/OR comparison](results/final_runs_v1/aggregate_phase11a/rf_ae_or_comparison.csv) and [detection overlap](results/final_runs_v1/aggregate_phase11a/rf_ae_overlap.csv). Preserve actual test FPR alongside recall: a validation FPR cap is not a test guarantee.

The Phase 11B results illustrate why contribution claims depend on operating point. At threshold 0.5, GATv2 mean test F1 is 0.996357 versus edge MLP 0.986736. At the validation-selected maximum-macro-F1 threshold, the paired mean F1 difference is -0.005954. Graph-Transformer-L8 minus L1 has mean F1 difference +0.002463 at 0.5 but -0.000276 at the validation 1% FPR criterion. These results do not establish consistent incremental benefit across operating points; see the linked summary for sample SD and all metrics.

Saved completion reports record independent verification of scores, metrics, cohort identities, threshold locks and checkpoints. Runtime measurements are CPU- and stage-specific; shared preparation and encoder costs must not be counted repeatedly or described as raw-data end-to-end serving latency. The [Phase 11B integrity report](results/final_runs_v1/aggregate_phase11b/integrity_report.md) retains the recovery-baseline hash limitations.

## Reproduction and artifact handling

Run commands from the repository root. Raw data, derived caches and Python environments remain local under ignored directories; a fresh clone requires reconstructed and verified prerequisites. The legacy `.venv` does not contain the modeling dependencies. Development used system Python with workspace runtimes in `data/phase4_runtime` and `data/temporal_runtime`; use the recorded environment for the relevant phase.

| Workflow | Instructions |
|---|---|
| Initial audit | [Historical workflow](docs/phase1_audit_history.md) |
| Phase 2 audit | [Verification and reproduction](results/phase2_audit_notes.md#verification-and-reproduction) |
| Frozen preparation | [Baseline pipeline](docs/BASELINE_PIPELINE.md) |
| Baselines and diagnostics | [Phase 4](docs/phase4_seed42.md), [Phase 4A](docs/phase4a_diagnostics.md) |
| Temporal development | [Phase 5](docs/phase5_temporal.md) |
| Graph development | [Phase 6](docs/phase6_graph.md) |
| Anomaly development | [Phase 7](docs/phase7_anomaly.md) |
| Fusion development | [Phase 8](docs/phase8_fusion.md) |
| Graph-temporal development | [Phase 9](docs/phase9_graph_temporal.md) |
| Final protocol and run plan | [Specification](results/final_spec_v1/FINAL_EXPERIMENT_SPEC.md), [run plan](results/final_spec_v1/final_run_plan.md) |
| Final execution implementation | [Phase 11A runner](src/final_runs/phase11a.py), [Phase 11B runner](src/final_runs/phase11b.py) |

For example, regenerate the temporal development report from saved outputs without fitting or inference:

```powershell
python -m src.temporal.report
```

Training runners may fit missing stages; an interrupted fit generally restarts its configuration rather than resuming optimizer/RNG state. Final execution runners contain audit and failure guards and are not generic rerun commands. Preserve completed artifacts and interruption/recovery records; do not rerun preparation into a frozen version. The final specification's original execution-authorization notice describes the Phase 10 freeze; later execution is recorded in the phase-specific completion reports.

`src/data/summarize_project.py` is a legacy Phase 4A README generator. Running it overwrites this README with the older phase map; it is not the regeneration command for this updated project overview.

## Remaining scope

Explainability/XAI (including SHAP), LSTM, a temporal autoencoder and end-to-end joint graph/Transformer/anomaly integration are outside the completed runs. No deployed online detector or general unseen-threat superiority is established. Any further experiment requires a separately versioned protocol that preserves the completed development and final-run evidence.
