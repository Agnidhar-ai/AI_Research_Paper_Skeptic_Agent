@echo off
cd /d "%~dp0"
title AI Research Paper Skeptic Agent
echo Starting AI Research Paper Skeptic Agent...
echo.
echo Keep this window open while using the app.
echo.
call ".venv\Scripts\activate.bat"
python -m streamlit run streamlit_app.py --server.port 8502 --server.address localhost --server.headless false
echo.
echo Streamlit stopped.
pause
