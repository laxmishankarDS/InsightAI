@echo off
title InsightAI Launcher

cd /d "%~dp0"

echo ==========================================
echo          INSIGHTAI PROJECT
echo ==========================================
echo.

echo [1/2] Starting FastAPI Backend...
start "InsightAI Backend" cmd /k "cd /d "%~dp0" && call .venv\Scripts\activate && cd project && python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

timeout /t 5 /nobreak >nul

echo [2/2] Opening InsightAI Dashboard...
start "" "%~dp0project\insightai_website\insightai.html"

echo.
echo ==========================================
echo Backend:  http://127.0.0.1:8000
echo Dashboard opened in browser
echo ==========================================
echo.
pause