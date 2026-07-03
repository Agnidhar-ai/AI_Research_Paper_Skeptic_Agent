@echo off
set PORT=%1
if "%PORT%"=="" set PORT=8502

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Create it first with: python -m venv .venv
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.port %PORT% --server.address localhost
