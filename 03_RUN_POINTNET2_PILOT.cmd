@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo  Gate E step 2: PointNet++ SSG pilot training (primary model)
echo  Requires GATE_D_ACCEPTANCE.json with the accepted generator digest.
echo  Without it this launcher exits 3 (GATE_D_NOT_ACCEPTED) and trains nothing.
echo ============================================================
python "%~dp0research\tree_qc\train_pointnet2.py" %*
set RC=%errorlevel%
echo.
if "%RC%"=="3" (
  echo POINTNET2_PILOT: REFUSED - GATE_D_NOT_ACCEPTED. No training was started.
) else if "%RC%"=="0" (
  echo POINTNET2_PILOT: completed.
) else (
  echo POINTNET2_PILOT: FAILED with exit code %RC%.
)
exit /b %RC%
