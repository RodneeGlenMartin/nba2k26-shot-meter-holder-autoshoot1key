@echo off
title k2 - staircase clock
cd /d "%~dp0"

rem The keyboard hook needs admin or K is never suppressed and the game
rem sees your real key-up instead of ours. Re-launch elevated if needed.
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

rem Fail loudly if the old tool is still running - both hook K and fight.
tasklist /fi "imagename eq python.exe" 2>nul | find /i "python.exe" >nul
if %errorlevel% equ 0 (
    echo.
    echo   WARNING: a python.exe is already running.
    echo   If that is k_meter.py, close it first - both hook the K key.
    echo.
    timeout /t 4 >nul
)

python -u k2_runtime.py
echo.
echo   k2 exited with code %errorlevel%
pause
