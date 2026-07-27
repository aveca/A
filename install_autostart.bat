@echo off
:: Ajoute Vision Agent au démarrage de Windows
set SCRIPT_DIR=%~dp0
set SHORTCUT_NAME=Vision Agent
set STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup

:: Créer le raccourci dans le dossier Démarrage
powershell -Command ^
  $ws = New-Object -ComObject WScript.Shell; ^
  $sc = $ws.CreateShortcut('%STARTUP_DIR%\%SHORTCUT_NAME%.lnk'); ^
  $sc.TargetPath = '%SCRIPT_DIR%run_vision_agent.bat'; ^
  $sc.WorkingDirectory = '%SCRIPT_DIR%'; ^
  $sc.Save()

echo Vision Agent ajoute au demarrage de Windows.
echo (Present dans : %STARTUP_DIR%)
pause
