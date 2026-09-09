@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0research\tree_qc\run_mini_gate.ps1"
exit /b %errorlevel%
