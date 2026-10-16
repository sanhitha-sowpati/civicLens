@echo off
setlocal
cd /d "%~dp0"
echo === CivicLens setup ===
python --version
if errorlevel 1 (
  echo Python is not installed or is not on PATH.
  pause
  exit /b 1
)
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed.
  pause
  exit /b 1
)
if not exist "artifacts\model.joblib" (
  echo Model artifacts missing. Training CatBoost once...
  python scripts\train.py
  if errorlevel 1 (
    echo Model training failed.
    pause
    exit /b 1
  )
)
echo Starting CivicLens at http://127.0.0.1:8000
python run_civiclens.py
pause
