# Windows Executable Build Instructions

This document provides step-by-step instructions for building a standalone Windows `.exe` for the PGG Cattle Monitoring Application.

## Prerequisites

### 1. Python Environment
- **Python 3.11** installed and added to PATH
- Virtual environment created and activated
- All dependencies installed from `requirements.txt`

### 2. Required Software (on target machines)
The generated .exe will require these to be pre-installed on the target Windows machine:
- **Intel RealSense SDK 2.0** (for RealSense D455i support)
  - Download: https://github.com/IntelRealSense/librealsense/releases
  - Install the Windows installer (.exe)
- **ZED SDK 4.0+** (for ZED 2i support)
  - Download: https://www.stereolabs.com/developers/release/
  - Install the Windows installer (.exe)

### 3. PyInstaller
Will be automatically installed by the build script, or install manually:
```bash
pip install pyinstaller
```

## Build Process

### Method 1: Automated Build (Recommended)

Simply run the batch script:

```cmd
build_windows_exe.bat
```

This will:
1. Check Python installation
2. Install PyInstaller
3. Clean previous builds
4. Generate the PyInstaller spec file
5. Build the executable
6. Verify the build

### Method 2: Manual Build

If you prefer to build manually:

```cmd
# Step 1: Generate spec file
python build_spec_generator.py

# Step 2: Build with PyInstaller
pyinstaller --clean --noconfirm pgg_app.spec
```

## Build Output

After successful build, you'll find:

```
dist/
└── PGG_Cattle_Monitor/
    ├── PGG_Cattle_Monitor.exe  <- Main executable
    ├── *.dll                    <- Required DLLs
    ├── *.pyd                    <- Python extension modules
    └── _internal/               <- Internal dependencies
```

## Distribution

To distribute the application:

1. **Copy the entire folder**: `dist/PGG_Cattle_Monitor/`
2. **Create installer** (optional): Use Inno Setup or NSIS to create a proper installer
3. **Include prerequisites**: Document that users must install RealSense SDK and ZED SDK

### Creating an Installer with Inno Setup

1. Download Inno Setup: https://jrsoftware.org/isinfo.php
2. Create an Inno Setup script (example provided below)
3. Compile to create `PGG_Cattle_Monitor_Setup.exe`

Example Inno Setup script (`installer.iss`):

```iss
[Setup]
AppName=PGG Cattle Monitor
AppVersion=1.0
DefaultDirName={pf}\PGG Cattle Monitor
DefaultGroupName=PGG Cattle Monitor
OutputDir=installer_output
OutputBaseFilename=PGG_Cattle_Monitor_Setup
Compression=lzma2
SolidCompression=yes

[Files]
Source: "dist\PGG_Cattle_Monitor\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{group}\PGG Cattle Monitor"; Filename: "{app}\PGG_Cattle_Monitor.exe"
Name: "{commondesktop}\PGG Cattle Monitor"; Filename: "{app}\PGG_Cattle_Monitor.exe"

[Run]
Filename: "{app}\PGG_Cattle_Monitor.exe"; Description: "Launch PGG Cattle Monitor"; Flags: postinstall nowait skipifsilent
```

## Troubleshooting

### Issue: "ModuleNotFoundError" when running .exe

**Solution**: Add missing module to hidden imports in `build_spec_generator.py`:
```python
hidden_imports = [
    # ... existing imports ...
    'missing_module_name',
]
```
Then rebuild.

### Issue: "DLL load failed"

**Solution**:
- Ensure RealSense SDK is installed on the target machine
- Ensure ZED SDK is installed on the target machine
- Check Windows Visual C++ Redistributables are installed

### Issue: Executable is too large (>500 MB)

**Solution**:
1. Exclude unnecessary modules in `build_spec_generator.py`:
```python
excludes = [
    'matplotlib',
    'IPython',
    'jupyter',
    # Add more unused modules
]
```

2. Use UPX compression (already enabled in spec file)

3. Consider using `--onefile` mode (slower startup, but single file):
```python
# In pgg_app.spec, change:
exe = EXE(
    ...
    exclude_binaries=False,  # Changed from True
    ...
)
# Remove COLLECT() section
```

### Issue: Slow startup time

**Cause**: PyInstaller unpacks files to temp directory on first run.

**Solutions**:
- Use `--onedir` mode (default, faster but more files)
- Optimize imports (remove unused heavy libraries)
- Pre-warm the application (run once after installation)

### Issue: "Failed to execute script" error

**Solution**:
1. Rebuild with console enabled for debugging:
```python
# In pgg_app.spec:
exe = EXE(
    ...
    console=True,  # Changed from False
    ...
)
```

2. Check error messages in console
3. Fix the issue
4. Rebuild with `console=False` for release

## Testing the Executable

Before distributing:

1. **Test on clean Windows VM**: Ensure it runs without development dependencies
2. **Test camera functionality**: Verify RealSense and ZED cameras are detected
3. **Test calibration**: Run multi-camera calibration workflow
4. **Test recording**: Verify synchronized recording from all cameras
5. **Test analytics**: Ensure cattle analytics modules work correctly
6. **Check file outputs**: Verify CSV exports, point clouds, videos are created

## Build Optimization Tips

### Reduce Build Size
```python
# In build_spec_generator.py, add to excludes:
excludes = [
    'matplotlib',
    'IPython',
    'jupyter',
    'sphinx',
    'pytest',
    'tkinter',  # If not using Tkinter
]
```

### Faster Builds
```cmd
# Use --noconfirm to skip confirmation prompts
pyinstaller --clean --noconfirm pgg_app.spec

# Skip UPX compression during development (faster build)
pyinstaller --clean --noconfirm --noupx pgg_app.spec
```

### Debug Mode
```python
# In pgg_app.spec for detailed error messages:
exe = EXE(
    ...
    debug=True,  # Enable debug mode
    console=True,  # Show console
    ...
)
```

## System Requirements for Target Machines

### Minimum Requirements
- Windows 10 64-bit or later
- 8 GB RAM
- Intel Core i5 or equivalent
- USB 3.0 ports for cameras
- Intel RealSense SDK 2.0+
- ZED SDK 4.0+ (if using ZED cameras)

### Recommended Requirements
- Windows 11 64-bit
- 16 GB RAM
- Intel Core i7 or equivalent
- USB 3.1 Gen 2 ports
- SSD storage for recording

## Support

For build issues:
1. Check PyInstaller documentation: https://pyinstaller.org/
2. Review error messages in console output
3. Check PyInstaller compatibility matrix for Python 3.11
4. Test with minimal spec file first, then add features incrementally

## Version History

- **v1.0** (2025-10-12): Initial build system
  - PyInstaller-based builds
  - Automated spec generation
  - Multi-camera support
  - Cattle analytics integration
