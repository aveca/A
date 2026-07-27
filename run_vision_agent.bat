@echo off
chcp 65001 >nul
echo ============================================
echo  Lancer Vision Agent (mode optimisé)
echo ============================================
echo.
set SCRIPT_DIR=%~dp0

set EXE="%SCRIPT_DIR%dist\ObjectRecognition\ObjectRecognition.exe"

if not exist %EXE% (
    echo ERREUR: executable non trouve.
    echo Lance build_exe.bat pour le generer.
    pause
    exit /b 1
)

echo Lancement en mode optimisé...
echo - Enregistrement activé
echo - OCR + YOLO + codes-barres activés
echo - VLM désactivé (plus rapide)
set VISION_DISABLE_VLM=1
start "" %EXE%

echo.
echo ============================================
echo  Lance en arriere-plan. Ouvre http://127.0.0.1:5000/app
echo ============================================
timeout /t 3 /nobreak >nul
start "" http://127.0.0.1:5000/app
