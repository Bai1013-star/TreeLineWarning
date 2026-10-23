$ErrorActionPreference = 'Stop'
try {
    Add-Type -AssemblyName System.Net.Http
    $taskGit = (Get-Command git.exe -ErrorAction Stop).Path
    $taskStart = New-Object Diagnostics.ProcessStartInfo
    $taskStart.FileName = $taskGit
    $taskStart.Arguments = 'credential fill'
    $taskStart.UseShellExecute = $false
    $taskStart.CreateNoWindow = $true
    $taskStart.RedirectStandardInput = $true
    $taskStart.RedirectStandardOutput = $true
    $taskStart.RedirectStandardError = $true
    $taskStart.EnvironmentVariables['GIT_TERMINAL_PROMPT'] = '0'
    $taskStart.EnvironmentVariables['GCM_INTERACTIVE'] = 'never'
    $taskProcess = New-Object Diagnostics.Process
    $taskProcess.StartInfo = $taskStart
    [void]$taskProcess.Start()
    $taskProcess.StandardInput.Write("protocol=https`nhost=github.com`npath=Bai1013-star/TreeLineWarning.git`n`n")
    $taskProcess.StandardInput.Close()
    $taskCredentialOutput = $taskProcess.StandardOutput.ReadToEnd()
    $taskCredentialError = $taskProcess.StandardError.ReadToEnd()
    $taskProcess.WaitForExit()
    if ($taskProcess.ExitCode -ne 0) { throw 'Sign in to this private repository with Git for Windows first, or download all release assets in your browser.' }
    $taskToken = $null
    foreach ($taskLine in ($taskCredentialOutput -split "`n")) {
        if ($taskLine -match '^password=(.+)$') { $taskToken = $Matches[1].TrimEnd("`r") }
    }
    $taskCredentialOutput = $null
    $taskCredentialError = $null
    if (-not $taskToken) { throw 'No GitHub credential is available for this private repository.' }
    $taskHandler = New-Object Net.Http.HttpClientHandler
    $taskHandler.AllowAutoRedirect = $false
    $taskClient = New-Object Net.Http.HttpClient($taskHandler)
    $taskClient.Timeout = [TimeSpan]::FromMinutes(30)
    $taskClient.DefaultRequestHeaders.UserAgent.ParseAdd('TreeLineWarning-Portable-Download')
    $taskApi = 'https://api.github.com/repos/Bai1013-star/TreeLineWarning'
    function Get-TaskResponse([string]$taskUrl, [string]$taskAccept, [bool]$taskAuthenticated) {
        $taskUri = [Uri]$taskUrl
        if ($taskUri.Scheme -ne 'https') { throw 'Only HTTPS downloads are permitted.' }
        $taskMessage = New-Object Net.Http.HttpRequestMessage([Net.Http.HttpMethod]::Get, $taskUrl)
        if ($taskAuthenticated) {
            if ($taskUri.Host -ne 'api.github.com') { throw 'Authentication must stay on the GitHub API host.' }
            $taskMessage.Headers.Authorization = New-Object Net.Http.Headers.AuthenticationHeaderValue('Bearer', $taskToken)
        } elseif (-not $taskUri.Host.EndsWith('.githubusercontent.com')) { throw 'Unexpected binary download host.' }
        $taskMessage.Headers.Accept.ParseAdd($taskAccept)
        try { return $taskClient.SendAsync($taskMessage, [Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult() }
        finally { $taskMessage.Dispose() }
    }
    $taskResponse = Get-TaskResponse ($taskApi + '/releases/420841750?fresh=' + [DateTime]::UtcNow.Ticks) 'application/vnd.github+json' $true
    try {
        [void]$taskResponse.EnsureSuccessStatusCode()
        $taskRelease = $taskResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json
    } finally { $taskResponse.Dispose() }
    if ($taskRelease.draft -or $taskRelease.tag_name -ne 'v2026.10.22') { throw 'The verified release is not published yet.' }
    $taskAssets = @{}
    foreach ($taskAsset in $taskRelease.assets) { $taskAssets[$taskAsset.name] = $taskAsset }
    $taskFolder = Join-Path $PSScriptRoot 'TreeLineWarning_20261022_download'
    New-Item -ItemType Directory -Path $taskFolder -Force | Out-Null
    function Save-TaskAsset([string]$taskName) {
        if ($taskName -notmatch '^(TreeLineWarning_20261022\.zip\.part[0-9]{3}|portable-manifest\.json|Restore-Portable\.ps1|restore-portable\.cmd)$') { throw 'Invalid asset name.' }
        $taskAsset = $taskAssets[$taskName]
        if (-not $taskAsset -or $taskAsset.state -ne 'uploaded' -or $taskAsset.digest -notmatch '^sha256:[a-f0-9]{64}$') { throw ('Missing verified asset: ' + $taskName) }
        $taskHash = $taskAsset.digest.Substring(7)
        $taskDestination = Join-Path $taskFolder $taskName
        if (Test-Path -LiteralPath $taskDestination) {
            if ((Get-FileHash -LiteralPath $taskDestination -Algorithm SHA256).Hash.ToLowerInvariant() -eq $taskHash) {
                Write-Host ('Already verified: ' + $taskName)
                return
            }
            throw ('An existing file differs. Move it before retrying: ' + $taskDestination)
        }
        for ($taskAttempt = 1; $taskAttempt -le 5; $taskAttempt++) {
            $taskTemporary = Join-Path $taskFolder ([Guid]::NewGuid().ToString('N') + '.download')
            $taskResponse = $null
            try {
                Write-Host ('Downloading: ' + $taskName)
                $taskResponse = Get-TaskResponse ($taskApi + '/releases/assets/' + $taskAsset.id + '?fresh=' + [DateTime]::UtcNow.Ticks) 'application/octet-stream' $true
                if ([int]$taskResponse.StatusCode -in @(301,302,303,307,308)) {
                    $taskLocation = $taskResponse.Headers.Location.AbsoluteUri
                    $taskResponse.Dispose()
                    $taskResponse = Get-TaskResponse $taskLocation 'application/octet-stream' $false
                }
                [void]$taskResponse.EnsureSuccessStatusCode()
                $taskFile = [IO.File]::Open($taskTemporary, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
                try { $taskResponse.Content.CopyToAsync($taskFile).GetAwaiter().GetResult() } finally { $taskFile.Dispose() }
                if ((Get-Item -LiteralPath $taskTemporary).Length -ne $taskAsset.size) { throw 'Download size mismatch.' }
                if ((Get-FileHash -LiteralPath $taskTemporary -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskHash) { throw 'Download SHA-256 mismatch.' }
                Move-Item -LiteralPath $taskTemporary -Destination $taskDestination
                return
            } catch {
                if (Test-Path -LiteralPath $taskTemporary) { Remove-Item -LiteralPath $taskTemporary -Force }
                if ($taskAttempt -eq 5) { throw }
                Write-Host ('Connection interrupted; retrying only ' + $taskName)
                Start-Sleep -Seconds 3
            } finally { if ($taskResponse) { $taskResponse.Dispose() } }
        }
    }
    Save-TaskAsset 'portable-manifest.json'
    $taskManifest = Get-Content -LiteralPath (Join-Path $taskFolder 'portable-manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($taskManifest.sha256 -ne '9d26d24481a59c1aad0fdd1317b707b2cc602d1e9ebbf78e96f2e01d76644a96' -or $taskManifest.parts.Count -ne 82) { throw 'Unexpected portable package manifest.' }
    foreach ($taskPart in $taskManifest.parts) {
        if ($taskAssets[$taskPart.name].digest -ne ('sha256:' + $taskPart.sha256)) { throw 'Manifest and GitHub digest differ.' }
        Save-TaskAsset $taskPart.name
    }
    Save-TaskAsset 'Restore-Portable.ps1'
    Save-TaskAsset 'restore-portable.cmd'
    $taskToken = $null
    $taskClient.Dispose()
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $taskFolder 'Restore-Portable.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Package restoration failed.' }
    Write-Host ('Complete. Extract the verified ZIP in: ' + $taskFolder)
    exit 0
} catch {
    $taskToken = $null
    Write-Error $_
    exit 1
}
