@echo off
chcp 65001 >nul
setlocal EnableExtensions
set "REPO_ROOT=E:\TreeLineWarning-fast"
set "PYTHON_EXE=E:\Miniforge\envs\treeline\python.exe"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist "%PYTHON_EXE%" (
    echo 未找到指定 Python 环境：%PYTHON_EXE%
    echo 请确认 E:\Miniforge\envs\treeline 环境仍然存在，然后重新双击运行。
    pause
    exit /b 1
)

if not exist "%REPO_ROOT%\input_las\" mkdir "%REPO_ROOT%\input_las"

echo.
echo 新 LAS FAST 一键研究演示（非认证）
echo 输入文件夹：%REPO_ROOT%\input_las
echo.
"%PYTHON_EXE%" "%REPO_ROOT%\scripts\run_new_las_demo.py" --input-dir "%REPO_ROOT%\input_las"
set "RUN_EXIT=%ERRORLEVEL%"

if not "%RUN_EXIT%"=="0" (
    echo.
    echo 运行未完成。请查看输出目录中的 run.log，并保留本窗口信息。
)
echo.
pause
exit /b %RUN_EXIT%
