$ErrorActionPreference = 'Stop'
try {
    $taskManifestPath = Join-Path $PSScriptRoot 'portable-manifest.json'
    $taskManifest = Get-Content -LiteralPath $taskManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($taskManifest.archive -notmatch '^TreeLineWarning_20261022_[^\\/]+\.zip$') { throw 'Unexpected archive name.' }
    if ($taskManifest.sha256 -ne '9d26d24481a59c1aad0fdd1317b707b2cc602d1e9ebbf78e96f2e01d76644a96') { throw 'Unexpected archive digest.' }
    $taskOutput = Join-Path $PSScriptRoot $taskManifest.archive
    if (Test-Path -LiteralPath $taskOutput) {
        if ((Get-FileHash -LiteralPath $taskOutput -Algorithm SHA256).Hash.ToLowerInvariant() -eq $taskManifest.sha256) {
            Write-Host 'The ZIP is already restored and verified. Extract it to start the application.'
            exit 0
        }
        throw 'An output file already exists but its hash differs. Move that file before retrying.'
    }
    if ($taskManifest.parts.Count -ne 2) { throw 'Two parts are required.' }
    $taskPartPaths = @()
    foreach ($taskPart in $taskManifest.parts) {
        if ($taskPart.name -notmatch '^TreeLineWarning_20261022\.zip\.part[12]$') { throw 'Invalid part name.' }
        $taskPartPath = Join-Path $PSScriptRoot $taskPart.name
        Write-Host ('Checking ' + $taskPart.name)
        if ((Get-Item -LiteralPath $taskPartPath).Length -ne $taskPart.bytes) { throw 'Part size mismatch.' }
        if ((Get-FileHash -LiteralPath $taskPartPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskPart.sha256) { throw 'Part SHA-256 mismatch.' }
        $taskPartPaths += $taskPartPath
    }
    $taskStream = [IO.File]::Open($taskOutput, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
    try {
        foreach ($taskPartPath in $taskPartPaths) {
            $taskInput = [IO.File]::OpenRead($taskPartPath)
            try { $taskInput.CopyTo($taskStream, 4194304) } finally { $taskInput.Dispose() }
        }
    } finally { $taskStream.Dispose() }
    if ((Get-Item -LiteralPath $taskOutput).Length -ne $taskManifest.bytes) { throw 'Final ZIP size mismatch.' }
    if ((Get-FileHash -LiteralPath $taskOutput -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskManifest.sha256) { throw 'Final ZIP SHA-256 mismatch.' }
    Write-Host ('Verified: ' + $taskOutput)
    Write-Host 'Extract the ZIP, then open the recommended CMD launcher inside the application folder.'
    exit 0
} catch {
    Write-Error $_
    exit 1
}
