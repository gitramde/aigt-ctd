param([ValidateSet('Prepare','Execute')][string]$Mode = 'Prepare')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonExe = 'C:\ProgramData\anaconda3\python.exe'
$activeRuns = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match 'src\.final_runs'
}
if ($activeRuns) { throw 'Another final-run process is active. Do not overlap runs.' }
if ($Mode -eq 'Prepare') {
    & $pythonExe -B -u -m src.final_runs.phase11b --prepare
} else {
    & $pythonExe -B -u -m src.final_runs.phase11b --execute
}
if ($LASTEXITCODE -ne 0) { throw "Phase11B $Mode failed. Preserve the audit and inspect the reported error." }
