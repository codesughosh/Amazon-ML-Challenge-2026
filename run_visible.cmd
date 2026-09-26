@echo off
title Amazon ML Challenge 2026 - India 70k
cd /d "X:\Amazon ML Challenge 2026"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
echo ================================================================
echo   Amazon ML Challenge 2026  -  run_floor  India  70k entities
echo   started %DATE% %TIME%
echo ================================================================
".venv\Scripts\python.exe" -u run_floor.py --country India --sample-s1 70000 --max-per-s1 50 --max-block 1500 --n-rare 8 --solo-df 4000 --trees 900
echo.
echo ================================================================
echo   finished %DATE% %TIME%
echo ================================================================
pause
