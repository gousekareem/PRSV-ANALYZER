@echo off
setlocal

cd /d %~dp0

if exist D:\prsv_env\Scripts\activate.bat (
    call D:\prsv_env\Scripts\activate.bat
) else (
    if not exist .venv (
        echo [INFO] Creating virtual environment in project folder...
        py -m venv .venv
    )
    call .venv\Scripts\activate.bat
)

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing project requirements...
pip install -r requirements.txt

echo [INFO] Starting PRSV Research Diagnostic System...
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

endlocal