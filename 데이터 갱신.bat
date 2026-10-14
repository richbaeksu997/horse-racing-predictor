@echo off
cd /d "%~dp0"
echo [1/3] Downloading new race results...
python collect.py
if errorlevel 1 goto done
echo [2/3] Retraining the model (about 1 minute)...
python train.py
if errorlevel 1 goto done
echo [3/3] Building the web page (docs folder)...
python build_site.py
echo.
echo Done. To update the phone page: open GitHub Desktop, Commit, then Push.
:done
pause
