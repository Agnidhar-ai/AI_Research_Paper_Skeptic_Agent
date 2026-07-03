@echo off
cd /d "%~dp0"
title AI Research Paper Skeptic Agent
echo Starting AI Research Paper Skeptic Agent...
echo.
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.port 8502 --server.address localhost --server.headless true
echo.
echo Streamlit stopped. Press any key to close this window.
pause > nul
