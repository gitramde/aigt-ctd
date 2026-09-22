# Phase 7: benign-trained anomaly branch

Run from the repository root using system Python and the existing isolated runtimes (`data/phase4_runtime`, `data/temporal_runtime`):

```powershell
python -m unittest src.anomaly.test_anomaly
python -m src.anomaly.run
```

The old `.venv` does not contain the modeling dependencies. Access to the installed PyTorch runtime may require execution outside the restricted sandbox on this Windows installation.

The runner verifies baseline_v1 raw/cleaned/membership/cache hashes and protects Phase 4/4A/5/6 artifacts. It trains two feed-forward autoencoder candidates on every benign TRAIN row, selecting checkpoints and architecture by benign-validation MSE only. Each executed epoch must visit exactly 10,885,643 benign records. No AE subsampling, representation learning on validation/test, or preprocessing refit occurs. Maximum epochs are four and patience two. The frozen preprocessing was originally fitted on all TRAIN classes; this deliberate reuse is not described as benign-only preprocessing.

Isolation Forest uses a documented 250,000-row seed-42 sample of benign TRAIN data, 100 trees and 256 rows per tree. Sample identities, population size and actual union of per-tree fitting rows are recorded. Both final models score all benign TRAIN rows for score distributions and the entire validation/test membership.

Protocol A uses only benign validation scores for empirical FPR caps 5%, 1%, 0.5% and 0.1%. Its model/threshold lock is written before malicious validation/test scoring. Protocol B subsequently uses scored validation records and labels for maximum macro-F1 and binary F1. B is explicitly label-aware and does not provide strict development-held-out Infilteration calibration. No test labels select a model or threshold. Raw reconstruction errors and negative Isolation Forest score_samples are not probabilities.

Completed matching stages can be reused. Interrupted fitting restarts that model; checkpoints from incomplete stages are not treated as final. To regenerate reports without fitting:

```powershell
python -m src.anomaly.report
```

Requested outputs are under `results/anomaly_v1`, with configs, models, saved scores, runtime/integrity manifests and figures in subfolders. `data/anomaly_v1` contains derived benign IDs and benign-validation feature cache. Conventional comparisons reuse full-cohort Phase 4 predictions and existing label-aware Phase 4A thresholds; no supervised model is retrained. These thresholds differ from Protocol-A benign-only calibration.

Only seed 42 is run. The optional temporal autoencoder is deferred, and the runner stops before graph+temporal integration, explainability or risk fusion.
