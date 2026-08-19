@echo off
chcp 65001 >nul
title TreeLineWarning 文件夹到 Excel (FOLDER_TO_EXCEL V1)
echo ============================================================
echo   TreeLineWarning  FOLDER_TO_EXCEL V1
echo   请把唯一的源点云放入 workspace\input\pointcloud\ 目录
echo ============================================================
cd /d "%~dp0"

set WORKSPACE=%~1
if "%WORKSPACE%"=="" set WORKSPACE=workspace

echo 工作区: %WORKSPACE%
echo 正在启动本地浏览器工作流...
echo   STEP 1  人工选择档距
echo   STEP 2  人工审核单木（删除 / 合并 / 撤销）
echo   STEP 3  最终结果 + Prediction_Input.xlsx
echo.

python scripts\run_folder_to_excel.py --workspace "%WORKSPACE%" %*

if errorlevel 1 (
  echo.
  echo 运行失败或已中断，请查看上方中文错误信息。
  echo 关键位置：workspace\01_span_selection / 02_tree_analysis / 03_export
  pause
  exit /b 1
)

echo.
echo 完成。最终结果位于 %WORKSPACE%\03_export\Prediction_Input.xlsx
pause
exit /b 0
