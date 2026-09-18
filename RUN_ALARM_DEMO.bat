@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Environment missing. Run scripts\setup_mvp.ps1 from PowerShell first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m affective_metacontrol.mvp --alarm-demo --gain 0.85 --alarm-gain 8.0
pause
