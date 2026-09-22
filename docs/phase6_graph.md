# Phase 6: Feb-20 graph contribution

The user approved the proposed 10:30/11:00 chronological boundaries after reviewing the pre-split timeline. `results/graph_v1/configs/split_approval.json` records that approval and the proposal hash; the historical pre-split proposal remains unchanged.

Run from the repository root using system Python and the existing isolated runtimes in `data/phase4_runtime` and `data/temporal_runtime`:

```powershell
python -m unittest src.graph.test_graph src.graph.test_pre_split
python -m src.graph.run
```

The runner verifies prior artifacts, reuses completed matching stages, fits missing configurations, freezes every validation-selected threshold before test inference, evaluates and writes reports. It runs seed 42 only. An interrupted model fitting stage restarts that configuration. This Windows environment may require execution outside the restricted sandbox to read installed PyTorch modules; the legacy `.venv` lacks the required packages.

The experiment uses a native PyTorch implementation of edge-aware GATv2, tested against a dense reference, with no additional dependency installation. The two primary architecture candidates use hidden dimension 64/dropout 0.1 and hidden dimension 128/dropout 0.2, both with two layers and four heads. One minute is predeclared primary; the selected architecture is held fixed for five- and ten-minute sensitivity runs. Maximum epochs are four, patience two, and all checkpoint/threshold selection uses validation only.

An edge MLP approximately matches the selected GATv2 parameter count and uses identical target flow vectors. A self-only GATv2 control retains node history but removes inter-node messages. Newly fitted logistic regression, Random Forest and XGBoost use the same target cohort. Phase 4 models are not reused because their training/preprocessing includes future Feb-20 data relative to this within-day split.

The causal graph protocol is in `configs/protocol.json`: targets start in the final 30 seconds of each minute; context stops strictly before the second-30 cutoff. Historical flow features are eligible only after flow completion. Each prefix stays within its calendar window and partition. Training target stride is 16, evaluation stride 4; all models and window sensitivities share targets. Full eligible target features and a nine-feature historical edge subset are transformed using preprocessing fitted on every approved training row. Node counts/bytes/protocol/port aggregates use all eligible history; message passing retains at most 8,192 recently completed edges. Raw IP indices are topology metadata, not features.

Generate reports from saved outputs with:

```powershell
python -m src.graph.report
```

Reports and CSVs live in `results/graph_v1`, with checkpoints, configs, saved scores and figures in subdirectories. Derived metadata and feature arrays live in `data/graph_v1`. Raw data, cleaned baseline data and prior-phase artifacts are protected by recorded hashes. Do not rerun `src.graph.pre_split` to change an approved proposal; create a new version for a genuinely different split or protocol.
