@echo off
title MedPulse AI - Medicine Recommendation System
echo ========================================================
echo Starting MedPulse AI Medicine Recommendation System...
echo ========================================================
cd /d "%~dp0"
python -m streamlit run app.py
pause
