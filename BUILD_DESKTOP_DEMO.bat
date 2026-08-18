@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pyinstaller.exe" (
  echo [TreeLineWarning] Install desktop requirements first: .venv\Scripts\python.exe -m pip install -r desktop-requirements.txt
  pause
  exit /b 2
)
".venv\Scripts\pyinstaller.exe" --noconfirm desktop_build.spec
if errorlevel 1 pause
