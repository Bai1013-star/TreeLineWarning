@echo off
rem ============================================================================
rem  TreeLineWarning — Tree QC Public Pilot (Gate D unlocked + E/F/G merged)
rem
rem  Double-click this file. It locates the verified Python/torch environment,
rem  sets UTF-8, runs the resumable orchestrator and keeps the window open with
rem  the final status, the report path, the run id and the verdict.
rem
rem  It performs NO training logic of its own and NEVER installs, upgrades or
rem  deletes anything:
rem    * no NVIDIA driver change
rem    * no CUDA toolkit install
rem    * no file deletion or disk cleanup
rem    * no container or Windows configuration change
rem
rem  Interrupted? Double-click it again: the run resumes from the last verified
rem  stage and never retrains a completed, digest-verified model.
rem ============================================================================
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
title TreeLineWarning Tree QC Public Pilot

set "REPO=%~dp0"
if "%REPO:~-1%"=="\" set "REPO=%REPO:~0,-1%"
set "ORCHESTRATOR=%REPO%\research\tree_qc\run_tree_qc_efg.py"

echo ==========================================================================
echo  TreeLineWarning Tree QC Public Pilot
echo ==========================================================================
echo  Repository   : %REPO%
echo  Orchestrator : %ORCHESTRATOR%
echo.

if not exist "%ORCHESTRATOR%" (
    echo FAILED: orchestrator not found at
    echo          %ORCHESTRATOR%
    echo         Run this file from inside the research worktree.
    goto :finish_fail
)

rem --- locate the verified training interpreter (existing env only) ----------
set "PYTHON="
set "CANDIDATE=D:\TreeQCTraining\venv\Scripts\python.exe"
if exist "%CANDIDATE%" (
    "%CANDIDATE%" -c "import torch" >nul 2>&1
    if !errorlevel! equ 0 set "PYTHON=%CANDIDATE%"
)
if not defined PYTHON (
    if exist "%REPO%\research\tree_qc\.venv\Scripts\python.exe" (
        "%REPO%\research\tree_qc\.venv\Scripts\python.exe" -c "import torch" >nul 2>&1
        if !errorlevel! equ 0 set "PYTHON=%REPO%\research\tree_qc\.venv\Scripts\python.exe"
    )
)
if not defined PYTHON (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined PYTHON (
            "%%P" -c "import torch" >nul 2>&1
            if !errorlevel! equ 0 set "PYTHON=%%P"
        )
    )
)

if not defined PYTHON (
    echo STOP_ENVIRONMENT_NOT_READY
    echo   No Python interpreter with a working PyTorch was found.
    echo   Checked:
    echo     D:\TreeQCTraining\venv\Scripts\python.exe
    echo     %REPO%\research\tree_qc\.venv\Scripts\python.exe
    echo     python.exe on PATH
    echo   This launcher never installs or upgrades the training environment.
    goto :finish_fail
)

echo  Python       : %PYTHON%
"%PYTHON%" -c "import sys,torch;print('  Python ver   :',sys.version.split()[0]);print('  Torch        :',torch.__version__);print('  CUDA ready   :',torch.cuda.is_available())"
echo  Run root     : D:\TreeQCTraining\public_pilot_v1
echo.
echo  Log          : D:\TreeQCTraining\public_pilot_v1\logs\autopilot.log
echo  Live status  : D:\TreeQCTraining\public_pilot_v1\CURRENT_STATUS.json
echo.
echo  This run is unattended and resumable. Close this window at any time and
echo  double-click this file again to continue from the last verified stage.
echo ==========================================================================
echo.

set "EXITCODE=0"
"%PYTHON%" "%ORCHESTRATOR%" --auto %*
set "EXITCODE=%errorlevel%"

echo.
echo ==========================================================================
echo  FINAL STATUS
echo ==========================================================================
"%PYTHON%" -c "import json,pathlib,sys;p=pathlib.Path(r'D:\TreeQCTraining\public_pilot_v1\CURRENT_STATUS.json');d=json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {};print('  RUN ID   :',d.get('run_id'));print('  UPDATED  :',d.get('updated_at'));print('  PHASE    :',(d.get('final') or {}).get('verdict') or (d.get('stop') or {}).get('code') or 'IN PROGRESS');print('  FREEZE   :',(d.get('freeze') or {}).get('sha256'));print('  REPORT   :',(d.get('final') or {}).get('report_json'));print('  MARKDOWN :',(d.get('final') or {}).get('report_md'));[print('  [%02d] %-26s %s'%(i+1,s['title'],s['status'])) for i,s in enumerate(d.get('stages') or [])];[print('  PROGRESS %s: %s/%s (%.1f ex/s, ETA %.0f min)'%(k,v.get('done'),v.get('total'),v.get('examples_per_second') or 0,(v.get('eta_seconds') or 0)/60)) for k,v in (d.get('progress') or {}).items()]"
if exist "D:\TreeQCTraining\public_pilot_v1\reports\EFG_STOP_REPORT.json" (
    echo.
    echo  STOP REPORT:
    echo    D:\TreeQCTraining\public_pilot_v1\reports\EFG_STOP_REPORT.json
)
echo.
echo  Orchestrator exit code: %EXITCODE%
echo   0 = completed, 2 = refused bounded options, 4 = run lock/state problem,
echo   5 = fail-closed STOP (see the stop report)
echo ==========================================================================

:finish_fail
echo.
echo Press any key to close this window...
pause >nul
endlocal
