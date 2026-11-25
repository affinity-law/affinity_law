@echo off
setlocal
pushd "%~dp0" || exit /b 1
if exist "4html\.runtime\venv\Scripts\python.exe" (
    "4html\.runtime\venv\Scripts\python.exe" -c "import sys; sys.exit(sys.version_info[:2] != (3, 11))" >nul 2>&1
    if not errorlevel 1 goto managed
)
where python >nul 2>&1
if errorlevel 1 goto launcher
python -c "import sys; sys.exit(sys.version_info[:2] != (3, 11))" >nul 2>&1
if not errorlevel 1 goto python
:launcher
py -3.11 -V >nul 2>&1
if not errorlevel 1 goto py
if exist "%LocalAppData%\Programs\Python\Python311\python.exe" goto installed
where uv >nul 2>&1
if not errorlevel 1 goto uv
where winget >nul 2>&1
if errorlevel 1 goto missing
choice /M "The workbench needs Python 3.11. Install it for your Windows account now"
if errorlevel 2 goto missing
winget install --exact --id Python.Python.3.11 --scope user --source winget
if errorlevel 1 goto missing
if exist "%LocalAppData%\Programs\Python\Python311\python.exe" goto installed
py -3.11 -V >nul 2>&1
if not errorlevel 1 goto py
:missing
echo Install 64-bit Python 3.11, then double-click Start_Workbench.cmd again.
goto failed
:managed
"4html\.runtime\venv\Scripts\python.exe" -B start_workbench.py %*
goto done
:python
python -B start_workbench.py %*
goto done
:py
py -3.11 -B start_workbench.py %*
goto done
:installed
"%LocalAppData%\Programs\Python\Python311\python.exe" -B start_workbench.py %*
goto done
:uv
uv run --no-project --python 3.11 -- start_workbench.py %*
:done
if errorlevel 1 goto failed
popd
exit /b 0
:failed
echo Setup or startup did not finish. See the message above.
popd
pause
exit /b 1
