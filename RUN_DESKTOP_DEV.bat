@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [TreeLineWarning] Missing .venv\Scripts\python.exe
  pause
  exit /b 2
)
".venv\Scripts\python.exe" -m desktop.app
if errorlevel 1 pause
