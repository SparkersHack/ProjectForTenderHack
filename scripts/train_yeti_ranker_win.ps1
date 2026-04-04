param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$pythonCandidates = @(
    (Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe"),
    (Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\python3.13.exe"),
    (Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\python.exe")
)

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
    try {
        $version = & $candidate -c "import catboost; print(catboost.__version__)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $pythonExe = $candidate
            break
        }
    } catch {
    }
}

if (-not $pythonExe) {
    Write-Error "Не найден Python 3.13 с установленным catboost. Проверьте Windows Python environment."
    exit 1
}

$scriptPath = Join-Path $projectRoot "scripts\train_yeti_ranker.py"
& $pythonExe $scriptPath @ScriptArgs
exit $LASTEXITCODE
