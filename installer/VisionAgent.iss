#define MyAppName "Vision Agent"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Vision Agent"
#define MyAppExeName "ObjectRecognition.exe"
#define MyAppAssocName MyAppName + " File"
#define MyAppAssocExt ".va"
#define MyAppAssocKey StringChange(MyAppAssocName, " ", "") + MyAppAssocExt

[Setup]
AppId={{E4B5C6D7-A8F9-1234-5678-901234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\Vision Agent
UninstallDisplayIcon={app}\favicon.ico
DefaultGroupName={#MyAppName}
OutputDir=..
OutputBaseFilename=VisionAgent_Setup
SetupIconFile=..\favicon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest

[Languages]
Name: "french"; MessagesFile: "compiler:Languages\French.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\ObjectRecognition\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\run_vision_agent.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\license.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\run_vision_agent.bat"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\run_vision_agent.bat"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\run_vision_agent.bat"; Description: "Lancer Vision Agent"; Flags: postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\products_db.json"
Type: filesandordirs; Name: "{app}\sessions"
Type: filesandordirs; Name: "{app}\reports"
