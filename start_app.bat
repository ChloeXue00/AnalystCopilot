@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo   Starting RAG Assistant...
echo   Browser will open at http://localhost:8501
echo   * Keep this window open (closing it stops the app) *
echo   * Press Ctrl+C or close window to stop *
echo.
streamlit run app.py
pause
