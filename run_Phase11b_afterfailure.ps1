# Phase 11B controlled continuation
# Preserves completed runs and continues only from the failed seed-123 L1 stage.
# Run from the aigt-ctd project root.

$ErrorActionPreference = "Stop"

$Python = "C:\ProgramData\anaconda3\python.exe"
$Module = "src.final_runs.phase11b"

function Run-Stage {
    param(
        [Parameter(Mandatory=$true)][ValidateSet("fit","evaluate")][string]$Stage,
        [Parameter(Mandatory=$true)][int]$Seed,
        [Parameter(Mandatory=$true)][string]$Group,
        [Parameter(Mandatory=$true)][string]$Model
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "START: $Stage seed=$Seed group=$Group model=$Model"
    Write-Host "============================================================"

    & $Python -B -u -m $Module `
        --stage $Stage `
        --seed $Seed `
        --group $Group `
        --model $Model

    if ($LASTEXITCODE -ne 0) {
        throw "FAILED: $Stage seed=$Seed group=$Group model=$Model. Stopping without continuing."
    }

    Write-Host "PASS: $Stage seed=$Seed group=$Group model=$Model"
}

function Run-Model {
    param(
        [Parameter(Mandatory=$true)][int]$Seed,
        [Parameter(Mandatory=$true)][string]$Group,
        [Parameter(Mandatory=$true)][string]$Model
    )

    Run-Stage -Stage "fit" -Seed $Seed -Group $Group -Model $Model
    Run-Stage -Stage "evaluate" -Seed $Seed -Group $Group -Model $Model
}

Write-Host "Phase 11B controlled continuation"
Write-Host "This script does NOT rerun completed seed-42 models or seed-123 Edge MLP/GATv2."
Write-Host "It starts at the previously failed seed-123 GAT-Transformer-L1 fit."
Write-Host ""

# ------------------------------------------------------------------
# Seed 123: resume exactly at the failed stage.
# Already completed and intentionally skipped:
#   edge_mlp
#   gatv2
# ------------------------------------------------------------------

Run-Model -Seed 123 -Group "graph_temporal" -Model "gat_transformer_L1"
Run-Model -Seed 123 -Group "graph_temporal" -Model "gat_transformer_L8"
Run-Model -Seed 123 -Group "graph"          -Model "gatv2_self_only"

# ------------------------------------------------------------------
# Remaining seeds: execute frozen Phase 11B ORDER exactly.
# ORDER from phase11b.py:
# edge_mlp -> gatv2 -> gat_transformer_L1 -> gat_transformer_L8
# -> gatv2_self_only
# ------------------------------------------------------------------

$RemainingSeeds = @(456, 789, 1024)

foreach ($Seed in $RemainingSeeds) {
    Run-Model -Seed $Seed -Group "graph"          -Model "edge_mlp"
    Run-Model -Seed $Seed -Group "graph"          -Model "gatv2"
    Run-Model -Seed $Seed -Group "graph_temporal" -Model "gat_transformer_L1"
    Run-Model -Seed $Seed -Group "graph_temporal" -Model "gat_transformer_L8"
    Run-Model -Seed $Seed -Group "graph"          -Model "gatv2_self_only"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "ALL REQUESTED PHASE 11B CONTINUATION STAGES COMPLETED"
Write-Host "============================================================"
Write-Host ""
Write-Host "Do not start another experimental phase yet."
Write-Host "Next: run/produce the frozen Phase 11B aggregate verification/report."