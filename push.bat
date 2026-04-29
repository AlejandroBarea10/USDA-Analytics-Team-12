@echo off
cd /d "%~dp0"

:: Check if git is initialized
if not exist ".git" (
    echo Initializing git...
    git init
    git branch -M main
    git remote add origin https://github.com/AlejandroBarea10/USDA-Analytics-Team-12.git
)

:: Push everything
git add -A
git commit -m "update %date% %time%"
git push -f origin main

echo.
echo ========================================
echo   DONE! Code pushed to GitHub.
echo   Streamlit will redeploy automatically.
echo ========================================
echo.
pause
