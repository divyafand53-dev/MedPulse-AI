@echo off
title MedPulse AI - REST API Server
echo ========================================================
echo Starting MedPulse AI REST API Server (Port 8000)...
echo ========================================================
cd /d "%~dp0"
python -m uvicorn api:app --reload --port 8000
pause
