@echo off
cd /d "%~dp0"
where py >nul 2>&1
if errorlevel 1 (
  python run_review.py
) else (
  py -3 run_review.py
)
pause
