@echo off
echo ========================================================
echo  Running AI Code Assistant Automated Tests
echo ========================================================

set "PYTHON_CMD=D:\Important Softwares\Python311\python.exe"
if not exist "%PYTHON_CMD%" (
    set "PYTHON_CMD=python"
)

cd /d "%~dp0backend"
"%PYTHON_CMD%" -m pytest tests -v
pause
