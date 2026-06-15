param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ForwardArgs
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$venvDir = Join-Path $Root ".venv"
$venvPython = Join-Path $venvDir "Scripts\\python.exe"

if (-not (Test-Path $venvPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $venvDir
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $venvDir
    } else {
        throw "Python 3 is required but was not found on PATH."
    }
}

& $venvPython -m pip install -U pip "setuptools<82" wheel
& $venvPython -m pip install -e .
try {
    & $venvPython -m pip install pycocotools typeguard
} catch {
    Write-Warning "pycocotools install failed; continuing without tiny-human evaluation support."
    & $venvPython -m pip install typeguard
}

& $venvPython -c "import torch; print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('gpu_count:', torch.cuda.device_count()); [print(f'gpu[{i}]:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi -L
}

& $venvPython (Join-Path $Root "examples\\visdrone_sfr\\run_sfrfull_dataset_suite.py") @ForwardArgs
