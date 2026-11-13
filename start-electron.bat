@echo off
setlocal
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo Node.js is required. Run setup-electron.bat first.
  exit /b 1
)
node "%~dp0scripts\electron.cjs" start %*
set "result=%errorlevel%"
if not "%result%"=="0" pause
exit /b %result%
