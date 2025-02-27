@echo off
for /f "tokens=5" %%i in ('nvidia-smi ^| findstr /R "^[ ]*[0-9]"') do taskkill /PID %%i /F
