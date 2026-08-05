@echo off
rem Launches the Pulse-Echo COF Studio in your default browser.
rem A Streamlit app must be started BY streamlit; running app.py with
rem plain python does nothing. Requires: pip install streamlit
cd /d "%~dp0"
python -m streamlit run gui\app.py --browser.gatherUsageStats false
