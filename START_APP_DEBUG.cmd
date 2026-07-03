@echo off
cd /d "%~dp0"
title AI Research Paper Skeptic Agent - Debug Start
echo Starting AI Research Paper Skeptic Agent with debug output...
echo Keep this window open.
echo.

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Virtual environment Python was not found at .venv\Scripts\python.exe
  echo Run: python -m venv .venv
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -c "import streamlit, streamlit_app; print('Imports OK. Streamlit', streamlit.__version__)"
if errorlevel 1 (
  echo.
  echo ERROR: The app failed to import. Copy the error above.
  pause
  exit /b 1
)

echo.
echo Starting server at http://localhost:8502
echo.
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.port 8502 --server.address 127.0.0.1 --server.headless false

echo.
echo Streamlit stopped or failed to start. Copy any error above.
pause
