@echo off
REM TreeLineWarning V5A3 - Ordered tower-selection launcher (PRIMARY workflow).
REM
REM Starts the Giro3D FULL-SOURCE page from the PRODUCTION PDAL writers.copc
REM full COPC (cache\giro3d_full_pdal\dataset\p3a_full_pdal.copc.laz), serves
REM the tower-review API against the CLEAN ordered project, and points bounded
REM local RAW membership segmentation at the existing RAW tile cache.
REM
REM The historical custom-builder full COPC remains available for comparison:
REM   --config poc_config_full.json --dataset-root cache\giro3d_full_source\dataset
REM
REM The protected source path is resolved by Python.  This file is ASCII-only
REM with CRLF line endings and valid REM/echo/command lines only.
REM
REM PRIMARY WORKFLOW: ordered manual tower selection, T001 T002 ... in
REM confirmation corridor order.  Automatic detection stays available as
REM collapsible OPTIONAL ASSISTANCE in the viewer.
REM
setlocal
cd /d "%~dp0"

if exist "cache\tower_ordered_project\project.json" goto :project_ready

echo [ordered-tower] ordered project missing - preparing via Python...
python scripts\prepare_tower_ordered_project.py
if errorlevel 1 goto :prepare_failed
goto :project_ready

:prepare_failed
echo [ordered-tower] ordered project preparation failed.
exit /b 1

:project_ready
echo [ordered-tower] starting ordered tower-selection server on http://127.0.0.1:8092/
python scripts\serve_p3a_giro3d.py ^
  --port 8092 ^
  --config poc_config_full_pdal.json ^
  --dataset-root cache\giro3d_full_pdal\dataset ^
  --tower-project-root cache\tower_ordered_project ^
  --tower-tile-cache cache\tower_raw_tiles ^
  --tower-operator owner

if errorlevel 1 goto :server_failed
goto :done

:server_failed
echo [ordered-tower] server exited with an error - see the message above.

:done
endlocal
