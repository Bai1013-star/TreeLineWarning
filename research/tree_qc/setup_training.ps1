$ErrorActionPreference = 'Stop'
$trainingPython = 'D:\TreeQCTraining\venv\Scripts\python.exe'
if ((Get-PSDrive D).Free -lt 45GB) { throw 'STOP_DISK_GUARD: reserve 10 GiB for install and retain 35 GiB' }
if (!(Test-Path -LiteralPath $trainingPython)) { throw 'Existing independent Python 3.11 training environment is required; do not rebuild automatically.' }
$env:TEMP='D:\TreeQCTraining\tmp'
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR='D:\TreeQCTraining\pip-cache'
& $trainingPython -c "import torch; assert torch.__version__ == '2.14.0+cu130'"
if ($LASTEXITCODE -eq 0) {
    Write-Output 'Selected official torch is already installed; no download needed. Run Gate A for actual CUDA validation.'
    exit 0
}
$wheel = 'D:\TreeQCTraining\gate_a_recovery_20260909\torch-2.14.0+cu130-cp311-cp311-win_amd64.whl'
$expectedHash = '00fb5440e7560b3b21e5e24567c6096fec4ef65b50a7ef1a105ebed5755eef7e'
if (!(Test-Path -LiteralPath $wheel)) {
    & $trainingPython "$PSScriptRoot\resume_official_torch_20260909.py"
    if ($LASTEXITCODE -ne 0) { throw 'Official wheel recovery did not complete. Preserve .part files and inspect download.jsonl before retrying.' }
}
if ((Get-FileHash -LiteralPath $wheel -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) { throw 'Official wheel SHA256 mismatch; installation refused.' }
& $trainingPython -m pip install $wheel --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw 'CUDA wheel install failed' }
& $trainingPython -m pip freeze | Set-Content D:\TreeQCTraining\logs\requirements-installed.txt
