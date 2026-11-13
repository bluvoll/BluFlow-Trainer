@echo off
setlocal
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo Node.js 22.12 or newer is required. Install Node.js and run this script again.
  exit /b 1
)
node "%~dp0scripts\electron.cjs" setup %*
set "result=%errorlevel%"
if not "%result%"=="0" pause
exit /b %result%
