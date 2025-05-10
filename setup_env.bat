@echo off
setlocal EnableDelayedExpansion

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo The following libraries will be checked/installed:
type requirements.txt

echo.
set /p user_input=Do you want to proceed? (Y/N): 
if /i "%user_input%"=="Y" (
    for /f "usebackq delims=" %%i in ("requirements.txt") do (
        set "lib=%%i"
        if "!lib!"=="" (
            rem skip empty lines
            continue
        )

        echo Checking !lib!...
        pip show !lib! >nul 2>&1
        if !errorlevel! == 0 (
            echo !lib! is already installed.
        ) else (
            echo Installing !lib!...
            pip install !lib!
        )
    )
    echo.
    echo All done.
) else (
    echo Operation cancelled.
)

pause
