$ErrorActionPreference = 'Stop'
$trainingPython = 'D:\TreeQCTraining\venv\Scripts\python.exe'
$runRoot = 'D:\TreeQCTraining\gate_a_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
New-Item -ItemType Directory -Force $runRoot | Out-Null
if (!(Test-Path $trainingPython)) {
    @{status='ENVIRONMENT_NOT_READY'; gate_a='NOT_RUN'; mini_overfit='NOT_RUN'; public_download_allowed=$false; error='Run SETUP_TREE_QC_TRAINING.cmd first; independent Python missing'} | ConvertTo-Json | Set-Content "$runRoot\mini_gate.json"
    Write-Output "Environment missing; evidence: $runRoot"
    exit 2
}
& $trainingPython "$PSScriptRoot\gate_preflight.py" --output $runRoot *> "$runRoot\mini_gate.raw.log"
$gateExit = $LASTEXITCODE
Get-Content "$runRoot\mini_gate.raw.log"
Write-Output "Gate A artifacts: $runRoot"
exit $gateExit
