@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Restore-Portable.ps1"
if errorlevel 1 (
  echo Restore failed. Read the error above before retrying.
  pause
  exit /b 1
)
pause
