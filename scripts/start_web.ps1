param(
    [string]$BindHost = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8010,
    [switch]$Factory,
    [switch]$Foreground,
    [switch]$Background,
    [switch]$AllowExternal,
    [string]$DatabasePath,
    [string]$TemporalModelPath,
    [string]$SpeciesGrowthModelPath,
    [string]$TrainingOutputDir,
    [string]$WebIndexPath,
    [string]$LogDir,
    [switch]$Stop,
    [switch]$DryRun
)

$projectRoot = Split-Path -Parent $PSScriptRoot
if ($env:OS -eq "Windows_NT") {
    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
} else {
    $venvPython = Join-Path $projectRoot ".venv/bin/python"
}

if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $pythonExecutable = $venvPython
} else {
    $pythonCommand = Get-Command python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $pythonCommand) {
        throw "未找到项目 .venv 或当前可执行 Python。"
    }
    $pythonExecutable = $pythonCommand.Source
}

$scriptPath = Join-Path $PSScriptRoot "start_web.py"
$pythonArguments = @('--host', $BindHost, '--port', $Port.ToString())

if ($Factory) { $pythonArguments += '--factory' }
if ($Foreground) { $pythonArguments += '--foreground' }
if ($Background) { $pythonArguments += '--background' }
if ($AllowExternal) { $pythonArguments += '--allow-external' }
if ($Stop) { $pythonArguments += '--stop' }
if ($DryRun) { $pythonArguments += '--dry-run' }

$optionalParameters = @{
    '--database-path' = $DatabasePath
    '--temporal-model-path' = $TemporalModelPath
    '--species-growth-model-path' = $SpeciesGrowthModelPath
    '--training-output-dir' = $TrainingOutputDir
    '--web-index-path' = $WebIndexPath
    '--log-dir' = $LogDir
}
foreach ($parameterName in $optionalParameters.Keys) {
    $value = $optionalParameters[$parameterName]
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        $pythonArguments += @($parameterName, $value)
    }
}

& $pythonExecutable $scriptPath @pythonArguments
exit $LASTEXITCODE
