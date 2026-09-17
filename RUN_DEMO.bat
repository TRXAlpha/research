@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Environment missing. Run scripts\setup_mvp.ps1 from PowerShell first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m affective_metacontrol.mvp --demo --max-new-tokens 64
pause

