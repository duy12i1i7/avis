param(
    [string]$RepoDir = $PSScriptRoot,
    [string]$Device = "0",
    [int]$Epochs = 300,
    [int]$Batch = 8,
    [int]$Imgsz = 960
)

$ErrorActionPreference = "Stop"
Set-Location $RepoDir

& git pull

& (Join-Path $RepoDir "run_sfr_full_suite.ps1") `
    --stage train `
    --device $Device `
    --epochs $Epochs `
    --batch $Batch `
    --imgsz $Imgsz

& (Join-Path $RepoDir "summarize_sfr_full_results.ps1") `
    --project-root "runs/sfr_full" `
    --output-csv "runs/sfr_full/sfrfull_summary.csv" `
    --output-md "runs/sfr_full/sfrfull_summary.md"

& (Join-Path $RepoDir "compare_sfr_full_vs_baseline.ps1") `
    --baseline-root "runs/sfr_suite" `
    --sfrfull-root "runs/sfr_full" `
    --output-csv "runs/sfr_full/sfrfull_vs_base.csv" `
    --output-md "runs/sfr_full/sfrfull_vs_base.md"
