@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo  Gate E step 4: cross-site held-out evaluation
echo  Wytham held-out pool = 749 trees (STRICT_HIGH_CONTAINMENT_QUARANTINE).
echo  The threshold must already be frozen from validation; held-out never
echo  selects a threshold, a hyper-parameter or a model.
echo  Requires GATE_D_ACCEPTANCE.json. Without it this exits 3 and runs nothing.
echo ============================================================
python "%~dp0research\tree_qc\run_tree_qc_efg.py" --dry-run
set RC=%errorlevel%
echo.
echo CROSS_SITE_EVAL: dry-run plan above. The held-out evaluation itself is
echo implemented in run_tree_qc_efg.py step 4 and remains locked until
echo Gate D acceptance plus a frozen validation threshold exist.
if "%RC%"=="0" exit /b 3
exit /b %RC%
