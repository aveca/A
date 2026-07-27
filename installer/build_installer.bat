@echo off
chcp 65001 >nul
echo ============================================
echo  Build de l'installateur Vision Agent
echo ============================================

set ISSCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist %ISSCC% (
    echo Inno Setup 6 non trouve a : %ISSCC%
    echo Telecharge-le depuis : https://jrsoftware.org/isdl.php
    pause
    exit /b 1
)

cd /d "%~dp0"
%ISSCC% "VisionAgent.iss"
if %errorlevel% neq 0 (
    echo ERREUR lors du build de l'installateur.
    pause
    exit /b 1
)

echo.
echo Installateur genere : VisionAgent_Setup.exe
echo.
pause
