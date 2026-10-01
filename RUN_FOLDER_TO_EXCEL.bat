@echo off
setlocal
title TreeLineWarning FOLDER_TO_EXCEL V1
echo ============================================================
echo   TreeLineWarning FOLDER_TO_EXCEL V1
echo   Put exactly one source LAS/LAZ into workspace\input\pointcloud\
echo ============================================================
cd /d "%~dp0"

set WORKSPACE=%~1
if "%WORKSPACE%"=="" set WORKSPACE=workspace

echo Workspace: %WORKSPACE%
echo Starting local browser workflow...
echo   STEP 1 select span
echo   STEP 2 review trees (delete / merge / undo)
echo   STEP 3 final result + user Excel workbook
echo.

python scripts\run_folder_to_excel.py --workspace "%WORKSPACE%"

if errorlevel 1 (
  echo.
  echo Run failed or was interrupted. See the Chinese error message above.
  echo Key locations: workspace\01_span_selection / 02_tree_analysis / 03_export
  pause
  exit /b 1
)

echo.
echo Done. Find the final XLSX in %WORKSPACE%\03_export\
pause
exit /b 0
