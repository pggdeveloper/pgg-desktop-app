; Inno Setup Script for PGG Cattle Monitoring Application
; This creates a professional Windows installer with Start Menu shortcuts,
; desktop icon, uninstaller, and proper file associations.
;
; Prerequisites:
;   - Download Inno Setup from https://jrsoftware.org/isinfo.php
;   - Build the .exe first using build_windows_exe.bat
;   - Run Inno Setup Compiler on this file
;
; Usage:
;   1. Open Inno Setup Compiler
;   2. File > Open > installer.iss
;   3. Build > Compile
;   4. Find installer in installer_output/PGG_Cattle_Monitor_Setup.exe

[Setup]
; Application information
AppName=PGG Cattle Monitor
AppVersion=1.0.0
AppPublisher=PGG
AppPublisherURL=https://www.pgg.com
AppSupportURL=https://www.pgg.com/support
AppUpdatesURL=https://www.pgg.com/updates
AppCopyright=Copyright (C) 2025 PGG

; Installation directories
DefaultDirName={autopf}\PGG Cattle Monitor
DefaultGroupName=PGG Cattle Monitor
DisableProgramGroupPage=yes

; Output
OutputDir=installer_output
OutputBaseFilename=PGG_Cattle_Monitor_Setup_v1.0.0
SetupIconFile=icons\app.ico
UninstallDisplayIcon={app}\PGG_Cattle_Monitor.exe

; Compression
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMADictionarySize=1048576
LZMANumFastBytes=273

; Privileges
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog

; Architecture
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

; Visual appearance
WizardStyle=modern
DisableWelcomePage=no
ShowLanguageDialog=no

; License and readme
LicenseFile=LICENSE.txt
InfoBeforeFile=INSTALL_README.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main application files
Source: "dist\PGG_Cattle_Monitor\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Documentation
Source: "BUILD_INSTRUCTIONS.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "CLAUDE.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}\docs"; Flags: ignoreversion isreadme

[Icons]
; Start menu shortcuts
Name: "{group}\PGG Cattle Monitor"; Filename: "{app}\PGG_Cattle_Monitor.exe"
Name: "{group}\Documentation"; Filename: "{app}\docs"
Name: "{group}\{cm:UninstallProgram,PGG Cattle Monitor}"; Filename: "{uninstallexe}"

; Desktop shortcut
Name: "{autodesktop}\PGG Cattle Monitor"; Filename: "{app}\PGG_Cattle_Monitor.exe"; Tasks: desktopicon

; Quick launch shortcut
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\PGG Cattle Monitor"; Filename: "{app}\PGG_Cattle_Monitor.exe"; Tasks: quicklaunchicon

[Run]
; Option to launch application after installation
Filename: "{app}\PGG_Cattle_Monitor.exe"; Description: "{cm:LaunchProgram,PGG Cattle Monitor}"; Flags: nowait postinstall skipifsilent

[Code]
// Custom installer code for checking prerequisites

function IsDotNetInstalled: Boolean;
begin
  // Check if .NET Framework or .NET Core is installed (if needed)
  Result := True; // Modify based on actual requirements
end;

function IsRealsenseSDKInstalled: Boolean;
var
  RealsensePath: String;
begin
  // Check if Intel RealSense SDK is installed
  Result := RegQueryStringValue(HKLM, 'SOFTWARE\Intel\RealSense', 'InstallPath', RealsensePath);
  if not Result then
    Result := RegQueryStringValue(HKLM64, 'SOFTWARE\Intel\RealSense', 'InstallPath', RealsensePath);
end;

function IsZEDSDKInstalled: Boolean;
var
  ZEDPath: String;
begin
  // Check if ZED SDK is installed
  Result := RegQueryStringValue(HKLM, 'SOFTWARE\Stereolabs\ZED', 'InstallPath', ZEDPath);
  if not Result then
    Result := RegQueryStringValue(HKLM64, 'SOFTWARE\Stereolabs\ZED', 'InstallPath', ZEDPath);
end;

function InitializeSetup(): Boolean;
var
  MissingPrereqs: String;
begin
  Result := True;
  MissingPrereqs := '';

  // Check for Intel RealSense SDK
  if not IsRealsenseSDKInstalled then
    MissingPrereqs := MissingPrereqs + '- Intel RealSense SDK 2.0' + #13#10;

  // Check for ZED SDK
  if not IsZEDSDKInstalled then
    MissingPrereqs := MissingPrereqs + '- Stereolabs ZED SDK 4.0+' + #13#10;

  // Show warning if prerequisites are missing
  if MissingPrereqs <> '' then
  begin
    if MsgBox('The following required software is not installed:' + #13#10#13#10 +
              MissingPrereqs + #13#10 +
              'The application may not work correctly without these components.' + #13#10#13#10 +
              'Do you want to continue installation anyway?',
              mbConfirmation, MB_YESNO) = IDNO then
    begin
      Result := False;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // Create output directories
    ForceDirectories(ExpandConstant('{app}\output'));
    ForceDirectories(ExpandConstant('{app}\snaps'));
    ForceDirectories(ExpandConstant('{app}\recordings'));
    ForceDirectories(ExpandConstant('{app}\calibration'));
  end;
end;

[UninstallDelete]
; Clean up generated files during uninstall
Type: filesandordirs; Name: "{app}\output"
Type: filesandordirs; Name: "{app}\snaps"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\logs"

[Messages]
; Custom messages
WelcomeLabel2=This will install [name/ver] on your computer.%n%nPGG Cattle Monitor is a comprehensive cattle monitoring system with multi-vendor camera integration for livestock analytics.%n%nIt is recommended that you close all other applications before continuing.
