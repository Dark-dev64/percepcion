@echo off
setlocal
pushd "%~dp0"
wsl -d Ubuntu -- bash ./scripts/setup_cpu.sh
set "result=%errorlevel%"
popd
exit /b %result%
