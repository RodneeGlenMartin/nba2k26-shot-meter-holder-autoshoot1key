@echo off
title K-Meter
cd /d "%~dp0"
echo.
echo    K-Meter - pick a build (never run both, they share the K hook)
echo.
echo    [1]  TEMPO   blocks K, pro stick down then flick up   (current)
echo    [2]  BUTTON  holds K and releases K                   (preserved)
echo.
choice /c 12 /n /m "   Which build? "
if errorlevel 2 goto button
if errorlevel 1 goto tempo

:tempo
call "%~dp0tempo\run_k_meter.bat"
goto :eof

:button
call "%~dp0button\run_k_meter.bat"
goto :eof
