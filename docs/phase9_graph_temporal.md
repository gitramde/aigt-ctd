# Phase 9 graph-temporal contribution development

Uses the exact Phase-6 Feb-20 cohort and frozen selected 1-minute GATv2 checkpoint. No earlier artifacts are changed. Temporal heads operate on concatenated graph source/destination embeddings and full target flow features. One historical slot represents an actual completed sampled flow from the corresponding previous calendar minute, with directed-pair identity preferred and role-consistent source/destination fallback. Missing minutes remain masked. Each current cutoff admits only historical flows completing strictly before that cutoff, within the same partition.

The L1 capacity control and L4/L8 temporal models use the same d_model=64, two-layer, four-head Transformer with dropout 0.1, FFN 128, and eight-slot position table. Only the heads are trained, for at most four epochs, seed42. All lengths are reported. Existing MLP/GAT controls are not retrained.

```powershell
python -B -m unittest src.graph_temporal.test_temporal
python -B -m src.graph_temporal.prepare
python -B -m src.graph_temporal.run
python -B -m src.graph_temporal.verify
```

System Python uses the existing isolated runtimes. Windows sandbox restrictions may require access outside the sandbox to read installed PyTorch. The runner reuses complete validated representation/model stages. Preparation computes coverage before fitting. The documented initial majority-complete L8 gate was revised during diagnostics, before training, because 74,150 fully populated TRAIN contexts support the requested masked-input experiment. Both the original gate and amendment are retained under results/graph_temporal_v1/configs. This is an explicit development decision, not a preregistered criterion.

Model inference timings distinguish frozen graph processing and temporal-head lookup/compute, and preserve original Phase-6 control measurements. This is an offline start-time-split benchmark, not strict online replay or unseen-threat detection. Stop after seed42 Phase9; no anomaly, fusion, explanation, or five-seed expansion.
