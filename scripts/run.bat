@echo off
cd /d "%~dp0\.."
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -m meeting_copilot
) else (
  python -m meeting_copilot
)
pause
