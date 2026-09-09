@echo off
if "%~1"=="" (
  "D:\TreeQCTraining\venv\Scripts\python.exe" "%~dp0research\tree_qc\zenodo_archive_recovery.py" ranges --chunk-mib 4
  exit /b
)
"D:\TreeQCTraining\venv\Scripts\python.exe" "%~dp0research\tree_qc\zenodo_archive_recovery.py" %*
exit /b %errorlevel%
