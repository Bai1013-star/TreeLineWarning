@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0research\tree_qc\setup_training.ps1"
exit /b %errorlevel%
