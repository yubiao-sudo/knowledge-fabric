@echo off
REM Pure ASCII only (see start.cmd for why).
cd /d "%~dp0"
python "%~dp0scripts\stop_all.py" %*
if errorlevel 1 pause