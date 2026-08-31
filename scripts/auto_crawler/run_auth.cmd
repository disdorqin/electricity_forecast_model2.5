@echo off
setlocal
cd /d "%~dp0"
pmos_auto_auth.exe
if errorlevel 1 pause
