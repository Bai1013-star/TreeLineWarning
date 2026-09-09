@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo  Gate E step 1: PointNet baseline training
echo  Requires GATE_D_ACCEPTANCE.json with the accepted generator digest.
echo  Without it this launcher exits 3 (GATE_D_NOT_ACCEPTED) and trains nothing.
echo ============================================================
python "%~dp0research\tree_qc\train_pointnet.py" %*
set RC=%errorlevel%
echo.
if "%RC%"=="3" (
  echo POINTNET_BASELINE: REFUSED - GATE_D_NOT_ACCEPTED. No training was started.
) else if "%RC%"=="0" (
  echo POINTNET_BASELINE: completed.
) else (
  echo POINTNET_BASELINE: FAILED with exit code %RC%.
)
exit /b %RC%
