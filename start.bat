@echo off
title FX Platform Launcher
echo ========================================
echo   Starting FX Forward-Test Platform
echo ========================================
echo.

:: Start Backend in a new terminal
echo Starting Backend (FastAPI on port 8000)...
start "FX Backend" cmd /k "cd /d %~dp0 && uvicorn fxbot.api:app --host 127.0.0.1 --port 8000 --reload"

:: Start Frontend in a new terminal
echo Starting Frontend (Vite on port 5173)...
start "FX Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Both services are launching in separate terminals.
echo   Backend:  http://127.0.0.1:8000
echo   Frontend: http://127.0.0.1:5173
echo.
pause
