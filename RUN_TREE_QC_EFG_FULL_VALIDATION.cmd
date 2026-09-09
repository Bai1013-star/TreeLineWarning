@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo  Gate E/F/G full validation orchestrator
echo  Steps: PointNet baseline -^> PointNet++ primary -^> validation threshold
echo         -^> held-out Wytham 749 blind eval -^> early-stop check
echo         -^> 3 seeds -^> 4096/2048 robustness -^> TREE_QC_EFG_FINAL_REPORT
echo  Requires GATE_D_ACCEPTANCE.json. Without it this exits 3 and runs nothing.
echo ============================================================
python "%~dp0research\tree_qc\run_tree_qc_efg.py" %*
set RC=%errorlevel%
echo.
if "%RC%"=="3" (
  echo FULL_EFG_VALIDATION: REFUSED - GATE_D_NOT_ACCEPTED. No training was started.
) else if "%RC%"=="0" (
  echo FULL_EFG_VALIDATION: plan resolved.
) else (
  echo FULL_EFG_VALIDATION: FAILED with exit code %RC%.
)
exit /b %RC%
