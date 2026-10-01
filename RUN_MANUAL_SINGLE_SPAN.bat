@echo off
chcp 65001 >nul
title 人工单档距分析 (MANUAL_SINGLE_SPAN)
echo ============================================================
echo  人工单档距分析 (MANUAL_SINGLE_SPAN)
echo  请稍候，正在启动本地浏览器界面...
echo ============================================================
cd /d "%~dp0"

python scripts\run_manual_single_span.py --input "input_las\2026-08-07_16没54-10_PointCloud_colorized.las" %*

if errorlevel 1 (
  echo.
  echo 运行失败或已中断，请查看上方错误信息。
  pause
  exit /b 1
)
echo.
echo 分析完成。结果位于 artifacts\manual_single_span_runs\ 下。
pause
exit /b 0
