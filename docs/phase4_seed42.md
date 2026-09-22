# Phase 4: seed-42 binary development baselines

The frozen baseline_v1 preparation must already be complete. No preparation files are rewritten. The Phase 4 preflight verifies raw, cleaned, membership and fitted-preprocessing hashes, reports actual membership counts, and checks baseline implementation hashes. Labels, identifiers and target-leaking columns remain excluded by the frozen feature specification.

Use the system Python 3.13.5, not the unrelated project .venv. Isolated dependencies live under data/phase4_runtime and are loaded by src.phase4.__init__. Install them if needed:

```powershell
python -m pip install --index-url https://pypi.org/simple --target data/phase4_runtime --no-deps --no-cache-dir xgboost==3.0.5 scikit-learn==1.7.2
python -m unittest src.phase4.test_phase4
python -m src.phase4.data
python -m src.phase4.run
```

Existing system dependencies: numpy 2.1.3, pandas 2.2.3, pyarrow 19.0.0, matplotlib, psutil, joblib and threadpoolctl. Model training JSON records the actual main library versions. The system sklearn used in Phase 3 is not overwritten. On this Windows installation, access to the installed isolated packages requires a process outside the restricted tool sandbox.

The training cache requires approximately 4.1 GB of disk and is a read-only float32 memory map during fitting. Cache creation first reproduces the original float64 transformed training checksum exactly. This is a model-input representation change, not refitted or modified preprocessing. Validation/test are transformed in batches with the same frozen pipeline. All 12,795,136 training rows are eligible; every SGD/MLP epoch visits every row. Random Forest uses 500,000 bootstrap draws per tree from the entire training pool. XGBoost builds a quantized matrix from batches. Workers run sequentially to bound peak memory.

The config is configs/phase4_seed42.json. Logistic Regression uses incremental SGD optimization of logistic loss with L2 regularization and parameter averaging. MLP uses incremental weighted Adam. Class weights are computed only from training counts; RF and LR receive class weights, MLP receives sample weights, and XGBoost receives scale_pos_weight. Optimizer order may shuffle inside training; membership and stored chronological order never change. Models use the same frozen scaled 80-column input, including the original fixed protocol encoding.

Validation macro-F1 selects checkpoints with a fixed threshold of 0.5. Selection histories record all candidate checkpoints. Two non-improving checkpoints stop fitting early. After all four training workers finish, a lock binds the selected model checksums and configuration before any test inference. Do not change configurations using test outcomes. This command runs seed 42 only; final multi-seed experiments are intentionally deferred.

Rerunning resumes at completed model/evaluation stage boundaries after hash checks. An interrupted model restarts its fitting stage; partially written checkpoints are not mistaken for completed models. To retry, use the same command. Do not delete or edit the frozen data or fitted preprocessing.

Outputs:
- models/: selected serialized models.
- metrics/: pretraining counts, selection histories, training metadata, configuration lock, saved probability arrays, evaluation JSON, requested metric/diagnostic/runtime CSVs, confusion matrix CSVs and final integrity audit.
- figures/: confusion matrix visualization directly from evaluation outputs.
- baseline_results_summary.md: generated findings and limitations.

ROC-AUC uses malicious probabilities. PR-AUC is the trapezoidal precision-recall area; average precision is additionally reported with its own name. Confusion matrices use actual rows and predicted columns, ordered benign then malicious. Missing attack families are explicitly absent with blank recall rather than zero recall. KNOWN/UNSEEN status comes only from training family presence. Infilteration and Bot have a dedicated diagnostic CSV. Binary detection is not multiclass recognition or proof of zero-day detection.

Training time includes data indexing and fitting; validation selection is separately timed. XGBoost training includes quantile-matrix construction. Inference end-to-end time includes streaming read and frozen transformation, while prediction-only time is also reported. Model loading and metric computation are excluded from inference time. Peak memory is the worker high-water RSS/Windows working set, including dependencies and arrays; serialized model bytes measure disk size. The same training measurement appears on both partition runtime rows.

