# Phase 8 frozen-score risk fusion

Run from the repository root with system Python and the existing isolated Phase-4 runtime:

```powershell
python -B -m unittest src.fusion.test_fusion
python -B -m src.fusion.run
python -B -m src.fusion.verify
```

The runner checks historical hashes and snapshots previous artifacts, reads saved RF/AE scores, calibrates and locks thresholds using validation, then evaluates the full frozen test membership. It does not load or refit models. Outputs and the exact protocol are in `results/fusion_v1`. All four primary score-fusion rules are reported without selecting a test winner.

OR uses the existing paired 1%, 0.5%, and 0.1% component thresholds. No saved RF 5% threshold exists. OR is explicitly a mixed-calibration comparator and does not inherit the component FPR cap. Its ROC/PR/AP use the binary decision, documented separately from continuous risk scores.

The critical comparison preserves the historical standalone operating points. Additional benign-only standalone RF thresholds appear in the full tables. Secondary maximum-macro-F1 thresholds are validation-label-aware. Source artifact hashes are checked again after analysis. The experiment stops at Phase 8.
