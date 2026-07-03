param(
    [int]$Port = 8502
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Set-Location $ProjectRoot

if (-not (Test-Path $PythonPath)) {
    Write-Error "Virtual environment not found. Create it first with: python -m venv .venv"
}

& $PythonPath -m streamlit run streamlit_app.py --server.port $Port --server.address localhost
