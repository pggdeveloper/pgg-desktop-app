# Quick Build Guide - Windows Executable

## TL;DR - Just Build It!

```cmd
# On Windows, just run:
build_windows_exe.bat

# Or if you prefer PowerShell:
.\build_windows_exe.ps1
```

That's it! Your `.exe` will be in `dist/PGG_Cattle_Monitor/`

---

## What You Need

1. **Windows 10/11** with Python 3.11 installed
2. **Virtual environment** activated with dependencies installed
3. That's it! PyInstaller will be installed automatically

## Build Options

### Option 1: Batch File (Easiest)
```cmd
build_windows_exe.bat
```

### Option 2: PowerShell (Prettier output)
```powershell
.\build_windows_exe.ps1
```

### Option 3: Manual (Full control)
```cmd
python build_spec_generator.py
pyinstaller --clean --noconfirm pgg_app.spec
```

## What Gets Created

```
dist/PGG_Cattle_Monitor/
├── PGG_Cattle_Monitor.exe    ← Your application!
├── *.dll                       ← System libraries
├── *.pyd                       ← Python extensions
└── _internal/                  ← Dependencies
```

## Distribution

### Quick Distribution
Just ZIP the entire `dist/PGG_Cattle_Monitor/` folder and send it!

### Professional Distribution
Use Inno Setup to create an installer:
1. Install [Inno Setup](https://jrsoftware.org/isinfo.php)
2. Open `installer.iss` in Inno Setup Compiler
3. Click Compile
4. Get `PGG_Cattle_Monitor_Setup.exe` in `installer_output/`

## Target Machine Requirements

Users need to install:
- **Intel RealSense SDK 2.0+** ([Download](https://github.com/IntelRealSense/librealsense/releases))
- **ZED SDK 4.0+** ([Download](https://www.stereolabs.com/developers/release/))
- Windows 10/11 64-bit

## Troubleshooting

### "ModuleNotFoundError"
→ Edit `build_spec_generator.py`, add missing module to `hidden_imports`, rebuild

### "DLL load failed"
→ User needs to install RealSense SDK and/or ZED SDK on their machine

### Exe is huge (>500 MB)
→ Normal for this app with OpenCV, Open3D, and camera SDKs. Can't reduce much.

### Slow startup
→ Normal on first run. Subsequent runs are faster.

### Need console for debugging
→ In `build_spec_generator.py`, change `console=False` to `console=True`, rebuild

## File Descriptions

| File | Purpose |
|------|---------|
| `build_windows_exe.bat` | Main build script (Batch) |
| `build_windows_exe.ps1` | Main build script (PowerShell) |
| `build_spec_generator.py` | Generates PyInstaller configuration |
| `pgg_app.spec` | PyInstaller config (auto-generated) |
| `installer.iss` | Inno Setup installer script |
| `BUILD_INSTRUCTIONS.md` | Detailed build documentation |
| `INSTALL_README.txt` | User installation guide |

## Pro Tips

**Development builds (faster):**
```cmd
pyinstaller --clean --noconfirm --noupx pgg_app.spec
```

**Debug mode:**
Edit `pgg_app.spec` → set `debug=True` and `console=True`

**Smaller exe (single file):**
Edit `pgg_app.spec` → set `exclude_binaries=False` in `EXE()`, remove `COLLECT()`

**Test on clean VM:**
Always test the final exe on a Windows machine without Python installed!

## Help

For detailed instructions: Read `BUILD_INSTRUCTIONS.md`

For user installation: Read `INSTALL_README.txt`

For build issues: Check PyInstaller logs in `build/` directory
