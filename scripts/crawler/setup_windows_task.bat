@echo off
REM ============================================================
REM  山东电力爬虫 — Windows 定时任务安装脚本
REM  以管理员身份运行一次即可
REM  任务名: "PMOS数据爬虫"
REM  执行时间: 每天 08:00 (避开14-16点申报时段)
REM ============================================================

echo ========================================
echo   安装 Windows 定时任务
echo   任务名: PMOS数据爬虫
echo   时间:   每天 08:00
echo ========================================
echo.

REM 获取当前脚本所在目录
set SCRIPTS_DIR=%~dp0
set PROJECT_DIR=%SCRIPTS_DIR%..\..

REM 创建定时任务（需要管理员权限）
schtasks /create ^
    /tn "PMOS数据爬虫" ^
    /tr "cmd /c cd /d %PROJECT_DIR% && python scripts\crawler\run_crawler.py" ^
    /sc daily ^
    /st 08:00 ^
    /f

if %ERRORLEVEL% equ 0 (
    echo.
    echo [OK] 定时任务创建成功！
    echo.
    echo 每天 08:00 自动执行: python scripts\crawler\run_crawler.py
    echo.
    echo 管理任务: 在 Windows 搜索"任务计划程序" ^|^> 任务计划程序库 ^|^> PMOS数据爬虫
) else (
    echo.
    echo [错误] 创建失败。请以管理员身份运行此脚本。
    echo.
    pause
    exit /b 1
)

pause
