@echo off
REM ============================================================================
REM Windows EXE Builder for PGG Cattle Monitoring Application
REM ============================================================================
REM This script automates the creation of a standalone Windows .exe
REM using PyInstaller. It handles all dependencies including camera SDKs.
REM
REM Prerequisites:
REM   - Python 3.11 installed
REM   - Virtual environment activated
REM   - All dependencies installed (pip install -r requirements.txt)
REM   - PyInstaller installed (pip install pyinstaller)
REM
REM Usage:
REM   build_windows_exe.bat
REM ============================================================================

echo.
echo ============================================================================
echo PGG Cattle Monitoring Application - Windows EXE Builder
echo ============================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.11 and add to PATH.
    pause
    exit /b 1
)

echo [1/6] Checking Python version...
python --version

REM Check if virtual environment is activated
python -c "import sys; sys.exit(0 if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Virtual environment not detected.
    echo It's recommended to activate your virtual environment first:
    echo   .venv\Scripts\activate
    echo.
    choice /C YN /M "Continue anyway"
    if errorlevel 2 exit /b 1
)

echo [2/6] Installing PyInstaller...
pip install pyinstaller --quiet
if errorlevel 1 (
    echo ERROR: Failed to install PyInstaller
    pause
    exit /b 1
)

echo [3/6] Cleaning previous build artifacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist __pycache__ rmdir /s /q __pycache__

echo [4/6] Generating PyInstaller spec file...
python build_spec_generator.py
if errorlevel 1 (
    echo ERROR: Failed to generate spec file
    pause
    exit /b 1
)

echo [5/6] Building Windows executable with PyInstaller...
echo This may take several minutes...
pyinstaller --clean --noconfirm pgg_app.spec
if errorlevel 1 (
    echo ERROR: PyInstaller build failed
    pause
    exit /b 1
)

echo [6/6] Post-build cleanup and verification...

REM Check if exe was created
if not exist "dist\PGG_Cattle_Monitor\PGG_Cattle_Monitor.exe" (
    echo ERROR: Executable was not created
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo BUILD SUCCESSFUL!
echo ============================================================================
echo.
echo The executable has been created in: dist\PGG_Cattle_Monitor\
echo.
echo Main executable: dist\PGG_Cattle_Monitor\PGG_Cattle_Monitor.exe
echo.
echo You can now:
echo   1. Test the executable by running it from dist\PGG_Cattle_Monitor\
echo   2. Distribute the entire dist\PGG_Cattle_Monitor\ folder
echo.
echo IMPORTANT NOTES:
echo   - Intel RealSense SDK must be installed on target machines
echo   - ZED SDK must be installed on target machines
echo   - First run may be slower due to DLL loading
echo.
echo ============================================================================
echo.

pause
