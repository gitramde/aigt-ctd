# Phase 4A reproducibility

Run `python -m src.phase4a.run` in the same isolated Phase 4 runtime. This performs only inference and diagnostic calculations. No model, split or preprocessing fitting occurs. Existing baseline files remain protected by SHA-256 checks; new outputs are confined to results/baseline_v1/diagnostics/.

Validation thresholds are persisted before transfer to test. The five objectives are maximum binary F1, maximum macro-F1, and maximum recall subject to validation FPR <= 1%, 0.5% or 0.1%. Ties favor fewer false positives and higher thresholds. Test threshold curves are descriptive only; they never select an operating point.

Training metrics use the whole training partition with the existing fitted models. Training probabilities are held in memory one model at a time; metrics and score checksums are persisted. Existing validation/test probabilities are reused unchanged. The final integrity check also verifies the original training cache hashes.

Feature summaries use every record and one column at a time. Numerical features retain native units and observed values before frozen imputation, with missing rates explicitly reported. Protocol is represented by its original frozen one-hot columns. Exact quantiles and KS statistics exclude missing values. PSI includes a missing bin and uses reference decile bins (unique cuts), unbounded tails and probability floor 1e-6 with renormalization. Constant references have a separate exact-value bin. Shift rankings use KS then PSI. Family rankings against both training populations use the minimum of the two KS values, preventing a large shift from only one reference from dominating.

Feature and training-inference checkpoints support resuming interrupted diagnostics. Re-running does not retrain; completed feature JSON records and training metrics are reused. Thresholds remain locked. Saved score diagnostics and plots can be regenerated. Use `python -m unittest src.phase4a.test_diagnostics` for synthetic threshold/KS/PSI checks.

All family, threshold, calibration, feature and reporting results are generated from experiment artifacts. The analysis explicitly distinguishes observed ranking from detection at 0.5, training performance from generalization, and empirical feature shift from a causal explanation. No known-family test recall is computed because test has no known malicious families.
