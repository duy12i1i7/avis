param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ForwardArgs
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$venvPython = Join-Path $Root ".venv\\Scripts\\python.exe"
if (Test-Path $venvPython) {
    & $venvPython (Join-Path $Root "examples\\visdrone_sfr\\summarize_sfrfull_results.py") @ForwardArgs
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 (Join-Path $Root "examples\\visdrone_sfr\\summarize_sfrfull_results.py") @ForwardArgs
} else {
    & python (Join-Path $Root "examples\\visdrone_sfr\\summarize_sfrfull_results.py") @ForwardArgs
}
