@echo off
title LiveCoach
cd /d "%~dp0"
python -u live_coach.py %*
pause
