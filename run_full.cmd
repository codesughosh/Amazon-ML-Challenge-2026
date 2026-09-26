@echo off
title Amazon ML Challenge 2026 - FULL TEST RUN
cd /d "X:\Amazon ML Challenge 2026"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
echo ================================================================
echo   FULL TEST INFERENCE  -  1,732,544 entities
echo   France -^> US -^> India  (writes output after each country)
echo   started %DATE% %TIME%
echo ================================================================
".venv\Scripts\python.exe" -u predict_test.py --train-sample 30000 --chunk 20000
echo.
echo   finished %DATE% %TIME%
pause
