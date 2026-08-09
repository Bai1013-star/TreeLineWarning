@echo off
chcp 65001 >nul
setlocal EnableExtensions
set "RUNS_ROOT=E:\TreeLineWarning-fast\artifacts\one_click_runs"

if not exist "%RUNS_ROOT%\" (
    echo 尚未找到一键运行结果文件夹。
    pause
    exit /b 1
)

for /f "delims=" %%D in ('dir /b /ad /o-d "%RUNS_ROOT%" 2^>nul') do (
    set "LATEST_RUN=%%D"
    goto :found
)

echo 尚未找到已完成的一键运行结果。
pause
exit /b 1

:found
set "SUMMARY=%RUNS_ROOT%\%LATEST_RUN%\07_report\RUN_SUMMARY.html"
if not exist "%SUMMARY%" (
    echo 最新结果没有 RUN_SUMMARY.html：%SUMMARY%
    pause
    exit /b 1
)

start "" "%SUMMARY%"
echo 已打开最新结果：%SUMMARY%
pause
exit /b 0
