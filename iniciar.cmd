@echo off
setlocal
pushd "%~dp0"
wsl -d Ubuntu -- bash ./scripts/start_cpu.sh %*
set "result=%errorlevel%"
popd
exit /b %result%
