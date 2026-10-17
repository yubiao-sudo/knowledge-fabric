@echo off
REM Pure ASCII only: cmd.exe parses .cmd with the system ANSI codepage,
REM so UTF-8 Chinese here would be mangled and break the batch parser.
REM All Chinese output comes from the Python script (UTF-8 safe).
cd /d "%~dp0"
python "%~dp0scripts\start_all.py" %*
if errorlevel 1 pause