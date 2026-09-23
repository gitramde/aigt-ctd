Set-Location 'C:\Users\ramku\Documents\codebase\aigt-ctd'
$py = 'C:\ProgramData\anaconda3\python.exe'
$ErrorActionPreference = 'Stop'

# Prevent overlapping training processes.
$running = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -match '^python' -and
        $_.CommandLine -match 'src\.final_runs'
    }
if ($running) {
    $running | Select-Object ProcessId, CommandLine
    throw 'An existing final-run process is active. Let it finish first.'
}

# Preserve existing stop records.
foreach ($path in @(
    'results/final_runs_v1/audit/STOP_FAILURE.json',
    'results/final_runs_v1/audit/phase11a/STOP_FAILURE.json'
)) {
    if (Test-Path $path) {
        throw "Review the recorded failure before continuing: $path"
    }
}

function Invoke-FinalStage {
    param([string[]]$StageArgs)
    & $py -B -u -m @StageArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Stage failed: $($StageArgs -join ' ')"
    }
}

# Verify existing completion records, checkpoints, scores and locks.
Invoke-FinalStage -StageArgs @(
    'src.final_runs.check_seed_provenance'
)
Invoke-FinalStage -StageArgs @(
    'src.final_runs.check_artifacts'
)
Invoke-FinalStage -StageArgs @(
    'src.final_runs.check_calibration'
)
Invoke-FinalStage -StageArgs @(
    'src.final_runs.check_semantics'
)

# Inspect the validated inventory without launching fits.
& $py -B -c "from src.final_runs.phase11a import inventory; import json; print(json.dumps(inventory(), indent=2))"
if ($LASTEXITCODE -ne 0) { throw 'Inventory verification failed.' }

# Complete only pending temporal and anomaly stages.
$jobs = @(
    @{ Group = 'temporal'; Model = 'transformer_L1' },
    @{ Group = 'temporal'; Model = 'transformer_L64' },
    @{ Group = 'anomaly';  Model = 'autoencoder_B' }
)

foreach ($seed in @(42, 123, 456, 789, 1024)) {
    foreach ($job in $jobs) {
        $group = $job.Group
        $model = $job.Model
        $base = "results/final_runs_v1/seed_$seed/$group"
        $training = "$base/metrics/${model}_training.json"
        $evaluation = "$base/metrics/${model}_evaluation.json"
        $checkpoint = "$base/models/$model.pt"
        $stageOptions = @(
            '--seed', "$seed", '--group', $group, '--model', $model
        )

        if (-not (Test-Path $training)) {
            if (Test-Path $checkpoint) {
                throw "Partial checkpoint requires reviewed recovery: $checkpoint"
            }
            Invoke-FinalStage -StageArgs (
                @('src.final_runs.worker') + $stageOptions
            )
        }

        if (-not (Test-Path $evaluation)) {
            Invoke-FinalStage -StageArgs (
                @('src.final_runs.evaluate') + $stageOptions
            )
        }
    }
}

# Aggregate and verify Phase 11A.
Invoke-FinalStage -StageArgs @('src.final_runs.report11a')
Invoke-FinalStage -StageArgs @('src.final_runs.check_artifacts')

Get-Content 'results/final_runs_v1/PHASE_11A_EXECUTION_SUMMARY.md'