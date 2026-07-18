@echo off
REM ============================================================
REM  山东电力爬虫 — Windows 定时任务安装脚本
REM  以管理员身份运行一次即可
REM  任务名: "PMOS数据爬虫"
REM  执行时间: 每天 08:00 (避开14-16点申报时段)
REM  特性: 到时自动唤醒睡眠中的电脑
REM ============================================================

echo ========================================
echo   安装 Windows 定时任务
echo   任务名: PMOS数据爬虫
echo   时间:   每天 08:00
echo   特性:   唤醒睡眠中的电脑执行
echo ========================================
echo.

REM 获取当前脚本所在目录
set SCRIPTS_DIR=%~dp0
set PROJECT_DIR=%SCRIPTS_DIR%..\..

REM 创建定时任务（需要管理员权限）
schtasks /create ^
    /tn "PMOS数据爬虫" ^
    /tr "cmd /c cd /d %PROJECT_DIR% && python scripts\crawler\run_crawler.py >> output\crawler_scheduled.log 2>&1" ^
    /sc daily ^
    /st 08:00 ^
    /du 00:30 ^
    /rl highest ^
    /f

if %ERRORLEVEL% equ 0 (
    echo.
    echo [OK] 定时任务创建成功！
    echo.
    echo 关键设置说明:
    echo   执行时间: 每天 08:00
    echo   超时限制: 30 分钟
    echo   日志输出: output\crawler_scheduled.log
    echo.
    echo ⚠ 重要: 请手动在任务计划程序中开启"唤醒计算机运行此任务"
    echo   操作步骤:
    echo     1. 在 Windows 搜索"任务计划程序"并打开
    echo     2. 左侧 → 任务计划程序库 → 找到 "PMOS数据爬虫"
    echo     3. 双击任务 → 条件 (Conditions) 选项卡
    echo     4. 勾选 "唤醒计算机运行此任务" (Wake the computer to run this task)
    echo     5. 确定
    echo.
    echo 提醒: 电脑需处于睡眠(休眠)状态，不能完全关机
    echo       每天下午 2-4 点申报时段请勿操作电脑
) else (
    echo.
    echo [错误] 创建失败。请以管理员身份运行此脚本。
    echo.
    pause
    exit /b 1
)

pause
