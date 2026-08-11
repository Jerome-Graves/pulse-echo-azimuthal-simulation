@echo off
rem Pulse-Echo Azimuthal Simulation - one-click GUI launcher.
rem
rem First run: downloads a private Python runtime into gui\runtime and
rem installs the required packages (needs internet, takes a few
rem minutes). Every run after that is offline and starts immediately.
rem
rem The GUI serves the studio pipeline and the sweep tab. Running new
rem forward simulations on GPU additionally needs CUDA and cupy in this
rem runtime; the analysis and visualisation layers do not.

setlocal
cd /d "%~dp0gui"

set "RUNTIME=%CD%\runtime"
set "PYTHON=%RUNTIME%\python\python.exe"

if exist "%PYTHON%" goto :run

echo ============================================================
echo  First-time setup: installing the GUI runtime.
echo  This needs an internet connection and a few minutes.
echo ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "setup_runtime.ps1"
if errorlevel 1 (
    echo.
    echo Setup failed. Check your internet connection and try again.
    pause
    exit /b 1
)

:run
rem Provide an icon-bearing shortcut next to this script (a .bat cannot
rem carry a custom icon itself).
if not exist "%~dp0Pulse-Echo Azimuthal Simulation.lnk" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "make_shortcut.ps1" >nul
)

echo Starting the simulation GUI (a browser tab will open shortly)...
cd /d "%~dp0"
"%PYTHON%" -m streamlit run gui\app.py --server.port 8552 --browser.gatherUsageStats false
if errorlevel 1 pause
