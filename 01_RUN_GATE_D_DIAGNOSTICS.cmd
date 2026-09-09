@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo  Gate D synthetic generator diagnostics (bounded, no training)
echo  Generates GATE_D_SYNTHETIC_DIAGNOSTIC.json,
echo  GATE_D_DIAGNOSTIC_MANIFEST.jsonl and representative visuals.
echo  This launcher never trains a model.
echo ============================================================
python "%~dp0research\tree_qc\gate_d_diagnostic.py" --per-partition 5 --repeats 1 %*
set RC=%errorlevel%
echo.
if "%RC%"=="0" (
  echo GATE_D_DIAGNOSTICS: PASS - no violations, no critical fingerprint.
) else (
  echo GATE_D_DIAGNOSTICS: HOLD - see research\tree_qc\reports\GATE_D_SYNTHETIC_DIAGNOSTIC.json
)
exit /b %RC%
