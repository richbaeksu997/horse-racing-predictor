@echo off
cd /d "%~dp0"
echo Open the address shown below on your phone (same Wi-Fi). Close this window to stop.
python app.py --lan
pause
