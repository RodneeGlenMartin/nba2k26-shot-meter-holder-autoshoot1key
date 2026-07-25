@echo off
title PrecisionTimer v3.1

REM --- Auto-elevate to Administrator (needed for key injection) ---
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

REM --- Run from this script's own folder ---
cd /d "%~dp0"

echo.
echo  ============================================
echo   PrecisionTimer  v3.1  (precision)
echo  ============================================
echo   Precision TIMER mode. TAP K once - no
echo   holding. The key is held for your exact
echo   time and released automatically.
echo   Tune with F6/F7 (-/+ 5ms).
echo  ============================================
echo.

python precision_timer.py

echo.
echo Script exited. Press any key to close.
pause >nul
