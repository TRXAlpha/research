$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ModelDir = Join-Path $ProjectRoot 'models\SmolLM2-1.7B-Instruct'

if (-not (Test-Path -LiteralPath $VenvPython)) {
    python -m venv (Join-Path $ProjectRoot '.venv')
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -e "$ProjectRoot[mvp]"

if (-not (Test-Path -LiteralPath (Join-Path $ModelDir 'config.json'))) {
    & $VenvPython -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolLM2-1.7B-Instruct', allow_patterns=['*.json','*.safetensors','tokenizer*','*.model','*.jinja'], local_dir=r'$ModelDir')"
}

Write-Host 'MVP setup complete.'
Write-Host "Run: $PSScriptRoot\run_mvp.ps1"
