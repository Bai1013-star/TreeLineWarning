@echo off
if "%~1"=="" (
  "D:\TreeQCTraining\venv\Scripts\python.exe" "%~dp0research\tree_qc\public_data.py" download --approval "%~dp0research\tree_qc\reports\PUBLIC_DATA_DOWNLOAD_APPROVAL.json"
  exit /b
)
"D:\TreeQCTraining\venv\Scripts\python.exe" "%~dp0research\tree_qc\public_data.py" %*
exit /b %errorlevel%
