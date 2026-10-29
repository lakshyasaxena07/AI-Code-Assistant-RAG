@echo off
echo ========================================================
echo  Starting AI Code Assistant (Local Development)
echo ========================================================

REM Set Python executable with fallback
set "PYTHON_CMD=D:\Important Softwares\Python311\python.exe"
if not exist "%PYTHON_CMD%" (
    set "PYTHON_CMD=python"
)

REM Check for backend .env
if not exist "%~dp0backend\.env" (
    copy "%~dp0backend\.env.example" "%~dp0backend\.env"
    echo [INFO] Created backend\.env from template.
)

REM 1. Start FastAPI Backend
echo [1/2] Starting FastAPI Backend on http://localhost:8000 ...
start "AI Code Assistant - Backend" cmd /k "cd /d "%~dp0backend" && "%PYTHON_CMD%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app"

REM Wait for backend initialization (loads PyTorch & embedding models on cold start)
echo Waiting for backend API to be ready...
set /a count=0
:wait_backend
set /a count+=1
curl.exe -s -f http://localhost:8000/health >nul 2>&1
if not errorlevel 1 goto backend_ready
if %count% geq 45 (
    echo [WARNING] Backend is taking longer than expected to initialize. Launching frontend anyway...
    goto backend_ready
)
ping -n 2 127.0.0.1 >nul
goto wait_backend

:backend_ready
echo [INFO] Backend API is ready!

REM 2. Start Streamlit Frontend
echo [2/2] Starting Streamlit Frontend on http://localhost:8501 ...
start "AI Code Assistant - Frontend" cmd /k "cd /d "%~dp0frontend" && "%PYTHON_CMD%" -m streamlit run app.py --server.port 8501"

echo ========================================================
echo  AI Code Assistant is launching!
echo  - Backend API:  http://localhost:8000
echo  - API Docs:     http://localhost:8000/docs
echo  - Frontend UI:  http://localhost:8501
echo ========================================================
