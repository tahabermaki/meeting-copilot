@echo off
cd /d "%~dp0\.."
echo Installation des dependances...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if not exist .env copy .env.example .env
echo.
echo Termine. Ouvre .env, colle tes deux cles API, puis lance scripts\run.bat
pause
