@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Environment missing. Run scripts\setup_mvp.ps1 from PowerShell first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m affective_metacontrol.neural_benchmark --output runs\neural-benchmark-1.7b.csv
pause

