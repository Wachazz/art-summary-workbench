@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
if errorlevel 1 (echo Install failed. & pause & exit /b 1)
python -m streamlit run streamlit_app.py
pause
