$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment not found: $python"
}

Push-Location $projectRoot
try {
    & $python -m PyInstaller --noconfirm --clean "TreeLineWarning.spec"
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed."
    }
    Write-Host "Created: $projectRoot\dist\TreeLineWarning.exe"
}
finally {
    Pop-Location
}
