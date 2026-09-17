$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw 'Virtual environment missing. Run scripts\setup_mvp.ps1 first.'
}

Push-Location $ProjectRoot
try {
    & $VenvPython -m affective_metacontrol.neural_benchmark @args
}
finally {
    Pop-Location
}

