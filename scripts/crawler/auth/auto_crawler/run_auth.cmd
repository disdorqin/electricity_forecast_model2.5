@echo off
setlocal
cd /d "%~dp0"
set "LOG_FILE=%~dp0crawler.log"
echo ================================================================
echo PMOS browser authentication
echo Log: %LOG_FILE%
echo ================================================================
echo [%date% %time%] start>> "%LOG_FILE%"
pmos_auto_auth.exe >> "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"
echo [%date% %time%] exit=%EXIT_CODE%>> "%LOG_FILE%"
type "%LOG_FILE%"
echo.
if not "%EXIT_CODE%"=="0" echo Authentication failed. Review crawler.log above.
pause
exit /b %EXIT_CODE%
