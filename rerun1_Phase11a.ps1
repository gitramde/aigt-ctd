Set-Location 'C:\Users\ramku\Documents\codebase\aigt-ctd'
$ErrorActionPreference = 'Stop'

# Stop if another final-run process is active.
$active = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -match '^python' -and
        $_.CommandLine -match 'src\.final_runs'
    }

if ($active) {
    $active | Select-Object ProcessId, CommandLine
    throw 'A final-run process is still active.'
}

$root = (Get-Location).Path
$temporal = Join-Path $root 'results\final_runs_v1\seed_42\temporal'
$audit = Join-Path $root 'results\final_runs_v1\audit\phase11a'

# Never archive a completed fit.
if (Test-Path "$temporal\metrics\transformer_L64_training.json") {
    throw 'L64 now has a completion record. Do not archive it.'
}

if (-not (Test-Path "$temporal\models\transformer_L64.pt")) {
    throw 'Expected partial checkpoint was not found.'
}

# Select only L64-specific files; preserve shared files and L1.
$files = @(
    Get-ChildItem -LiteralPath $temporal -Recurse -File |
        Where-Object { $_.Name -like 'transformer_L64*' }

    Get-ChildItem -LiteralPath $audit -File |
        Where-Object {
            $_.Name -like '*seed_42*model_transformer_L64.log'
        }
)

$archive = Join-Path $audit (
    'recovery_seed42_L64_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
)

# Validate every source and destination before moving anything.
foreach ($file in $files) {
    if (-not $file.FullName.StartsWith(
        $root + '\', [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Source outside workspace: $($file.FullName)"
    }
}
if (-not $archive.StartsWith(
    $root + '\', [StringComparison]::OrdinalIgnoreCase
)) {
    throw 'Archive is outside the workspace.'
}

New-Item -ItemType Directory -Path $archive | Out-Null

# Preserve original paths and hashes.
$manifest = foreach ($file in $files) {
    [pscustomobject]@{
        OriginalPath = $file.FullName
        RelativePath = $file.FullName.Substring($root.Length + 1)
        SHA256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    }
}
$manifest | Export-Csv "$archive\manifest.csv" -NoTypeInformation

@'
Interrupted seed-42 Transformer-L64 fit preserved before restarting.
No training-completion record exists.
Restart uses the same frozen configuration and seed from initialization.
Completed Transformer-L1 and shared artifacts remain in place.
'@ | Set-Content "$archive\recovery_note.txt"

foreach ($entry in $manifest) {
    $destination = Join-Path $archive $entry.RelativePath
    New-Item -ItemType Directory -Force `
        -Path (Split-Path $destination -Parent) | Out-Null

    Move-Item -LiteralPath $entry.OriginalPath -Destination $destination

    if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash `
        -ne $entry.SHA256) {
        throw "Archive verification failed: $destination"
    }
}

Write-Host "Partial L64 attempt preserved in: $archive"

# Restart pending fits sequentially.
.\run_phase11a_pending.ps1