# Phase 5 temporal development

Use the system Python and the existing workspace runtimes in `data/phase4_runtime` and `data/temporal_runtime`; the legacy `.venv` does not contain these dependencies. The saved training JSON files record package versions. Run from the repository root:

```powershell
python -m unittest src.temporal.test_temporal
python -m src.temporal.run
```

The runner verifies frozen hashes, reuses completed matching stages, trains only missing configurations, locks validation-selected operating thresholds, evaluates the selected Transformer and L1 control, and generates reports. An interrupted fitting stage restarts that configuration. Existing protocol/configuration mismatches fail rather than overwrite prior experiments.

To regenerate reports from saved outputs without training or inference:

```powershell
python -m src.temporal.report
```

Outputs are under `results/temporal_v1/`, including the generated `phase5_results_summary.md`, sequence/integrity reports, all requested CSVs, saved predictions, checkpoints, threshold lock and figures. Baseline comparisons index existing Phase 4 scores on exactly the same targets and reselect the 1% FPR operating point using only matched validation records.

The deliberately bounded development protocol uses seed 42, four temporal configurations, a three-epoch ceiling, training stride 128, evaluation stride 8 and a common 63-record daily warmup. Validation dominated by unseen Infilteration represents later-period generalization. The L1 control shares model capacity and targets. No final multi-seed runs or graph-based components are included. Test results have already been observed in previous phases, so this is development evidence, not a fresh confirmatory test.
