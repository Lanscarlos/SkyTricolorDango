@echo off
rem 双击启动管理面板（浏览器会自动打开 http://127.0.0.1:19390/）。
rem 这个窗口就是面板本身：可以最小化，关掉它面板就停了。
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)
"%PY%" -m skydango console %*
echo.
echo 管理面板已经退出。按任意键关闭这个窗口。
pause >nul
