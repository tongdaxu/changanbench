param(
    [switch]$InstallDeps,
    [switch]$SkipXiph,
    [int]$XiphLimit = 5
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$ScriptArgs = @("scripts/run_smoke.py", "--xiph-limit", "$XiphLimit")
if ($InstallDeps) {
    $ScriptArgs += "--install-deps"
}
if ($SkipXiph) {
    $ScriptArgs += "--skip-xiph"
}

if ($env:PYTHON) {
    & $env:PYTHON @ScriptArgs
    exit $LASTEXITCODE
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    & python @ScriptArgs
    exit $LASTEXITCODE
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 @ScriptArgs
    exit $LASTEXITCODE
}

Write-Error "Python was not found. Install Python 3 or set the PYTHON environment variable."
