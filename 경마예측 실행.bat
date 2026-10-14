@echo off
cd /d "%~dp0"
echo Starting... the browser will open in a few seconds. Close this window to stop.
start "" cmd /c "timeout /t 5 >nul & start http://127.0.0.1:5000"
python app.py
pause
