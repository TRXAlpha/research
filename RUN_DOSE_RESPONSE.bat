@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Environment missing. Run scripts\setup_mvp.ps1 from PowerShell first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m affective_metacontrol.dose_response --gain 0.65 --alarm-gain 2.5 --max-norm-ratio 0.30 --compare-unbounded
pause
