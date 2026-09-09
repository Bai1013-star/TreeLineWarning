$ErrorActionPreference = 'Stop'
if ((Get-PSDrive D).Free -lt 45GB) { throw 'STOP_DISK_GUARD: reserve 10 GiB for install and retain 35 GiB' }
New-Item -ItemType Directory -Force D:\TreeQCTraining\tmp,D:\TreeQCTraining\pip-cache,D:\TreeQCTraining\logs | Out-Null
$env:TEMP='D:\TreeQCTraining\tmp'
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR='D:\TreeQCTraining\pip-cache'
if (!(Test-Path D:\TreeQCTraining\venv\Scripts\python.exe)) {
    py -3.11 -m venv D:\TreeQCTraining\venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 required' }
}
& D:\TreeQCTraining\venv\Scripts\python.exe -m pip install 'torch==2.14.0+cu130' --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw 'CUDA wheel install failed' }
& D:\TreeQCTraining\venv\Scripts\python.exe -m pip freeze | Set-Content D:\TreeQCTraining\logs\requirements-installed.txt
