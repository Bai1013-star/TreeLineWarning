@echo off
chcp 65001 >nul
setlocal EnableExtensions DisableDelayedExpansion
set "REPO_ROOT=%~dp0"
set "PYTHON_EXE=E:\Miniforge\envs\treeline\python.exe"

if not exist "%PYTHON_EXE%" goto python_missing

"%PYTHON_EXE%" "%REPO_ROOT%scripts\run_new_las_demo.py"
set "RUN_EXIT=%ERRORLEVEL%"
if errorlevel 1 goto failed
exit /b 0

:python_missing
echo Python environment not found: E:\Miniforge\envs\treeline\python.exe
set "RUN_EXIT=1"
goto failed

:failed
echo.
echo Launcher failed. See run.log in the generated run folder if one was created.
if defined TREE_LINE_DEMO_NO_PAUSE exit /b %RUN_EXIT%
pause
exit /b %RUN_EXIT%
