# ============================================================================
# Windows EXE Builder for PGG Cattle Monitoring Application (PowerShell)
# ============================================================================
# This PowerShell script automates the creation of a standalone Windows .exe
# using PyInstaller. It handles all dependencies including camera SDKs.
#
# Prerequisites:
#   - Python 3.11 installed
#   - Virtual environment activated
#   - All dependencies installed (pip install -r requirements.txt)
#   - PyInstaller will be installed automatically
#
# Usage:
#   .\build_windows_exe.ps1
#
# Or if execution policy prevents running:
#   powershell -ExecutionPolicy Bypass -File build_windows_exe.ps1
# ============================================================================

# Set error action preference
$ErrorActionPreference = "Stop"

# Color output functions
function Write-Header {
    param([string]$Message)
    Write-Host "`n============================================================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "============================================================================`n" -ForegroundColor Cyan
}

function Write-Step {
    param([string]$Step, [string]$Message)
    Write-Host "[$Step] " -NoNewline -ForegroundColor Yellow
    Write-Host $Message -ForegroundColor White
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ " -NoNewline -ForegroundColor Green
    Write-Host $Message -ForegroundColor Green
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "✗ " -NoNewline -ForegroundColor Red
    Write-Host $Message -ForegroundColor Red
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "⚠ " -NoNewline -ForegroundColor Yellow
    Write-Host $Message -ForegroundColor Yellow
}

# Main build script
try {
    Write-Header "PGG Cattle Monitoring Application - Windows EXE Builder"

    # Check Python installation
    Write-Step "1/6" "Checking Python installation..."
    try {
        $pythonVersion = python --version 2>&1
        Write-Success "Python found: $pythonVersion"
    }
    catch {
        Write-Error-Custom "Python not found. Please install Python 3.11 and add to PATH."
        exit 1
    }

    # Check if virtual environment is activated
    Write-Step "2/6" "Checking virtual environment..."
    $venvActive = $false
    if ($env:VIRTUAL_ENV) {
        Write-Success "Virtual environment active: $env:VIRTUAL_ENV"
        $venvActive = $true
    }
    else {
        Write-Warning-Custom "Virtual environment not detected."
        Write-Host "It's recommended to activate your virtual environment first:" -ForegroundColor Yellow
        Write-Host "  .venv\Scripts\Activate.ps1" -ForegroundColor Yellow
        Write-Host ""
        $response = Read-Host "Continue anyway? (y/n)"
        if ($response -ne 'y' -and $response -ne 'Y') {
            Write-Host "Build cancelled." -ForegroundColor Yellow
            exit 0
        }
    }

    # Install PyInstaller
    Write-Step "3/6" "Installing PyInstaller..."
    try {
        pip install pyinstaller --quiet --disable-pip-version-check
        Write-Success "PyInstaller installed successfully"
    }
    catch {
        Write-Error-Custom "Failed to install PyInstaller"
        throw
    }

    # Clean previous build artifacts
    Write-Step "4/6" "Cleaning previous build artifacts..."
    $dirsToClean = @("build", "dist", "__pycache__")
    foreach ($dir in $dirsToClean) {
        if (Test-Path $dir) {
            Remove-Item -Recurse -Force $dir
            Write-Host "  Removed: $dir" -ForegroundColor DarkGray
        }
    }
    Write-Success "Cleanup complete"

    # Generate spec file
    Write-Step "5/6" "Generating PyInstaller spec file..."
    try {
        python build_spec_generator.py
        if ($LASTEXITCODE -ne 0) {
            throw "Spec generator returned error code $LASTEXITCODE"
        }
        Write-Success "Spec file generated successfully"
    }
    catch {
        Write-Error-Custom "Failed to generate spec file"
        throw
    }

    # Build executable
    Write-Step "6/6" "Building Windows executable with PyInstaller..."
    Write-Host "  This may take several minutes..." -ForegroundColor DarkGray
    Write-Host ""

    try {
        pyinstaller --clean --noconfirm pgg_app.spec
        if ($LASTEXITCODE -ne 0) {
            throw "PyInstaller returned error code $LASTEXITCODE"
        }
    }
    catch {
        Write-Error-Custom "PyInstaller build failed"
        throw
    }

    # Verify build
    $exePath = "dist\PGG_Cattle_Monitor\PGG_Cattle_Monitor.exe"
    if (Test-Path $exePath) {
        $exeSize = (Get-Item $exePath).Length / 1MB
        Write-Success "Executable created successfully"
        Write-Host "  Size: $([math]::Round($exeSize, 2)) MB" -ForegroundColor DarkGray
    }
    else {
        Write-Error-Custom "Executable was not created"
        exit 1
    }

    # Success message
    Write-Header "BUILD SUCCESSFUL!"
    Write-Host "The executable has been created in: " -NoNewline
    Write-Host "dist\PGG_Cattle_Monitor\" -ForegroundColor Green
    Write-Host ""
    Write-Host "Main executable: " -NoNewline
    Write-Host "dist\PGG_Cattle_Monitor\PGG_Cattle_Monitor.exe" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now:" -ForegroundColor Cyan
    Write-Host "  1. Test the executable by running it from dist\PGG_Cattle_Monitor\" -ForegroundColor White
    Write-Host "  2. Distribute the entire dist\PGG_Cattle_Monitor\ folder" -ForegroundColor White
    Write-Host "  3. Create an installer using Inno Setup (installer.iss)" -ForegroundColor White
    Write-Host ""
    Write-Host "IMPORTANT NOTES:" -ForegroundColor Yellow
    Write-Host "  - Intel RealSense SDK must be installed on target machines" -ForegroundColor White
    Write-Host "  - ZED SDK must be installed on target machines" -ForegroundColor White
    Write-Host "  - First run may be slower due to DLL loading" -ForegroundColor White
    Write-Host ""
    Write-Host "============================================================================`n" -ForegroundColor Cyan

    # Ask if user wants to test the executable
    $testNow = Read-Host "Do you want to test the executable now? (y/n)"
    if ($testNow -eq 'y' -or $testNow -eq 'Y') {
        Write-Host "`nLaunching executable..." -ForegroundColor Cyan
        Start-Process -FilePath $exePath
    }

}
catch {
    Write-Host "`n" -NoNewline
    Write-Error-Custom "Build failed with error:"
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "Stack trace:" -ForegroundColor DarkGray
    Write-Host $_.ScriptStackTrace -ForegroundColor DarkGray
    exit 1
}

# Pause at the end (if running in interactive mode)
if ($Host.Name -eq "ConsoleHost") {
    Write-Host "`nPress any key to continue..." -ForegroundColor DarkGray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
