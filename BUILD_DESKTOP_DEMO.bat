@echo off
setlocal
cd /d "%~dp0"
if not defined TREELINE_BUILD_PYTHON set "TREELINE_BUILD_PYTHON=.venv311\Scripts\python.exe"
if not exist "%TREELINE_BUILD_PYTHON%" (
  echo [TreeLineWarning] CPython 3.11 build environment is required. Set TREELINE_BUILD_PYTHON or create .venv311.
  pause
  exit /b 2
)
"%TREELINE_BUILD_PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"
if errorlevel 1 (
  echo [TreeLineWarning] CPython 3.11 is required for the bundled PDAL and Pillow extensions.
  pause
  exit /b 2
)
"%TREELINE_BUILD_PYTHON%" -m PyInstaller --noconfirm desktop_build.spec
if errorlevel 1 pause
