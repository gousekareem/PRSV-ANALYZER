@echo off
setlocal

cd /d %~dp0\prsv_project

if not exist .venv (
    echo [INFO] Creating virtual environment in prsv_project\.venv ...
    py -m venv .venv
)
call .venv\Scripts\activate.bat

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing project requirements...
pip install -r requirements.txt

if not exist .env (
    echo [INFO] Creating .env from .env.example...
    copy .env.example .env >nul
)

if not exist models\svm_model.joblib (
    echo [WARNING] No trained model found in models\. The app will run using
    echo [WARNING] a heuristic fallback instead of a real classifier.
    echo [WARNING] To train the real model, run in a separate terminal:
    echo [WARNING]   python scripts\auto_label_from_filenames.py
    echo [WARNING]   python scripts\retrain_model.py
)

echo [INFO] Starting PRSV Research Diagnostic System...
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

endlocal