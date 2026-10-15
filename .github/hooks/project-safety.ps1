$ErrorActionPreference = 'Stop'
$rawInput = [Console]::In.ReadToEnd()
if ([string]::IsNullOrWhiteSpace($rawInput)) {
    exit 0
}

try {
    $event = $rawInput | ConvertFrom-Json
} catch {
    [Console]::Error.WriteLine('Project safety hook could not parse the tool event.')
    exit 2
}

$toolName = [string]$event.toolName
if ([string]::IsNullOrEmpty($toolName)) {
    $toolName = [string]$event.tool_name
}
$inputObject = $event.toolArgs
if ($null -eq $inputObject) {
    $inputObject = $event.tool_input
}
$payload = if ($null -eq $inputObject) {
    ''
} else {
    $inputObject | ConvertTo-Json -Depth 64 -Compress
}
$isTerminal = $toolName -match '(?i)(bash|powershell|terminal|command|shell)'
$isFileWriter = $toolName -match '(?i)(edit|write|create|delete|move|rename|patch)'
$isMutatingCommand = $payload -match '(?i)(>\s*|Set-Content|Add-Content|Out-File|Remove-Item|Move-Item|Copy-Item|Export-Csv|to_csv|write_text|unlink|rmtree|shutil\.(copy|move)|open\s*\([^)]*,\s*["''][wax])'
$decisionReason = $null

if (($isFileWriter -or ($isTerminal -and $isMutatingCommand)) -and
    $payload -match '(?i)data[\\/]+raw[\\/]') {
    $decisionReason = 'This operation appears to modify immutable data/raw inputs. Confirm an explicit user authorization before proceeding.'
} elseif (($isFileWriter -or ($isTerminal -and $isMutatingCommand)) -and
    $payload -match '(?i)experiments[\\/][^"''\s,}]*_results\.json') {
    $decisionReason = 'This operation appears to overwrite an experiment results JSON. Require explicit user authorization and preserve prior results.'
} elseif ($isTerminal -and
    $payload -match '(?i)(python(?:core)?[-_]?3[._-]?(?!12(?:[\\/]|$))\d{1,2}[\\/]|python3\.(?!12\b)\d{1,2}|py(?:thon)?\s+-3\.(?!12\b)\d+|\.venv[\\/]+Scripts[\\/]+activate)') {
    $decisionReason = 'This command may switch away from the required Python 3.12 environment. Verify the interpreter and obtain explicit user authorization.'
}

if ($null -ne $decisionReason) {
    @{ permissionDecision = 'ask'; permissionDecisionReason = $decisionReason } |
        ConvertTo-Json -Compress |
        Write-Output
}
