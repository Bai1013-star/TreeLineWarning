@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Download-Portable.ps1"
if errorlevel 1 (
  echo Download failed. Verified files are kept so you can retry later.
  pause
  exit /b 1
)
pause
