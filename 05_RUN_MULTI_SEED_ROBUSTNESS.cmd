@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo  Gate E step 7: 4096 vs 2048 sampling robustness
echo  Runs the primary model at both point counts with the same frozen
echo  threshold logic. 4096 is the primary input; 2048 is the robustness mode.
echo  Requires GATE_D_ACCEPTANCE.json. Without it this exits 3 and runs nothing.
echo ============================================================
python "%~dp0research\tree_qc\run_tree_qc_efg.py" --dry-run
set RC=%errorlevel%
echo.
echo MULTI_SEED_ROBUSTNESS: dry-run plan above. The seed sweep and the 4096/2048
echo comparison are implemented in run_tree_qc_efg.py steps 6-7 and remain locked
echo until Gate D acceptance.
if "%RC%"=="0" exit /b 3
exit /b %RC%
