```bat
@echo off
setlocal EnableDelayedExpansion

echo ==================================================
echo Fake News Detection System - Automated Runner
echo ==================================================
echo.

:: Configuration
set VENV_DIR=venv
set PYTHON=python
set APP_FILE=app.py
set REQUIREMENTS=requirements.txt
set DEFAULT_MODE=gui
:: Available modes: gui, api, test-groq, train, retrain

:: Check if Python is installed
echo Checking for Python...
%PYTHON% --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not installed or not found in PATH.
    echo Please install Python 3.8+ and ensure it's added to PATH.
    pause
    exit /b 1
)

:: Check if virtual environment exists
echo Checking for virtual environment...
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment in %VENV_DIR%...
    %PYTHON% -m venv %VENV_DIR%
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate virtual environment
echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

:: Check and install dependencies
if exist "%REQUIREMENTS%" (
    echo Checking dependencies from %REQUIREMENTS%...
    for /f "usebackq delims=" %%i in ("%REQUIREMENTS%") do (
        set "lib=%%i"
        if "!lib!"=="" (
            rem Skip empty lines
            continue
        )
        echo Checking !lib!...
        pip show !lib! >nul 2>&1
        if !ERRORLEVEL% neq 0 (
            echo Installing !lib!...
            pip install !lib!
            if !ERRORLEVEL% neq 0 (
                echo ERROR: Failed to install !lib!.
                pause
                exit /b 1
            )
        )
    )
) else (
    echo WARNING: %REQUIREMENTS% not found. Assuming dependencies are installed.
)

:: Prompt user for run mode
echo.
echo Available run modes:
echo 1. GUI (default)
echo 2. API Server
echo 3. Test Groq API
echo 4. Train Models
echo 5. Force Retrain Models
echo.
set /p USER_MODE=Select mode (1-5, default is 1): 
if "!USER_MODE!"=="" set USER_MODE=1

:: Map user input to mode
if "!USER_MODE!"=="1" (
    set RUN_MODE=%DEFAULT_MODE%
    set ARGS=
) else if "!USER_MODE!"=="2" (
    set RUN_MODE=api
    set ARGS=--api
) else if "!USER_MODE!"=="3" (
    set RUN_MODE=test-groq
    set ARGS=--test-groq
) else if "!USER_MODE!"=="4" (
    set RUN_MODE=train
    set ARGS=--train
) else if "!USER_MODE!"=="5" (
    set RUN_MODE=retrain
    set ARGS=--force-retrain
) else (
    echo ERROR: Invalid mode selected. Using default (GUI).
    set RUN_MODE=%DEFAULT_MODE%
    set ARGS=
)

:: Check if app file exists
if not exist "%APP_FILE%" (
    echo ERROR: %APP_FILE% not found in current directory.
    pause
    exit /b поруш

System: It looks like the batch file content was cut off with an incomplete line (`exit /b`). I'll provide a complete and polished version of the `run_app.bat` batch file that automates running the Fake News Detection System. This version includes error handling, dependency checks, virtual environment setup, and user prompts for selecting the run mode. It also ensures the program runs smoothly in either GUI, API, or other modes.

<xaiArtifact artifact_id="0a2ef912-43fe-4396-93c2-ee8ed7ea2afc" artifact_version_id="aa756ebf-0257-40d3-9ef9-762cea10425f" title="run_app.bat" contentType="application/bat">
```bat
@echo off
setlocal EnableDelayedExpansion

echo ==================================================
echo Fake News Detection System - Automated Runner
echo ==================================================
echo.

:: Configuration
set VENV_DIR=venv
set PYTHON=python
set APP_FILE=app.py
set REQUIREMENTS=requirements.txt
set DEFAULT_MODE=gui
:: Available modes: gui, api, test-groq, train, retrain

:: Check if Python is installed
echo Checking for Python...
%PYTHON% --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not installed or not found in PATH.
    echo Please install Python 3.8+ and ensure it's added to PATH.
    pause
    exit /b 1
)

:: Check if virtual environment exists
echo Checking for virtual environment...
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment in %VENV_DIR%...
    %PYTHON% -m venv %VENV_DIR%
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate virtual environment
echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

:: Check and install dependencies
if exist "%REQUIREMENTS%" (
    echo Checking dependencies from %REQUIREMENTS%...
    for /f "usebackq delims=" %%i in ("%REQUIREMENTS%") do (
        set "lib=%%i"
        if "!lib!"=="" (
            rem Skip empty lines
            continue
        )
        echo Checking !lib!...
        pip show !lib! >nul 2>&1
        if !ERRORLEVEL% neq 0 (
            echo Installing !lib!...
            pip install !lib!
            if !ERRORLEVEL% neq 0 (
                echo ERROR: Failed to install !lib!.
                pause
                exit /b 1
            )
        ) else (
            echo !lib! is already installed.
        )
    )
) else (
    echo WARNING: %REQUIREMENTS% not found. Assuming dependencies are installed.
)

:: Prompt user for run mode
echo.
echo Available run modes:
echo 1. GUI (default)
echo 2. API Server
echo 3. Test Groq API
echo 4. Train Models
echo 5. Force Retrain Models
echo.
set /p USER_MODE=Select mode (1-5, default is 1): 
if "!USER_MODE!"=="" set USER_MODE=1

:: Map user input to mode
if "!USER_MODE!"=="1" (
    set RUN_MODE=%DEFAULT_MODE%
    set ARGS=
) else if "!USER_MODE!"=="2" (
    set RUN_MODE=api
    set ARGS=--api
) else if "!USER_MODE!"=="3" (
    set RUN_MODE=test-groq
    set ARGS=--test-groq
) else if "!USER_MODE!"=="4" (
    set RUN_MODE=train
    set ARGS=--train
) else if "!USER_MODE!"=="5" (
    set RUN_MODE=retrain
    set ARGS=--force-retrain
) else (
    echo WARNING: Invalid mode selected. Using default (GUI).
    set RUN_MODE=%DEFAULT_MODE%
    set ARGS=
)

:: Check if app file exists
if not exist "%APP_FILE%" (
    echo ERROR: %APP_FILE% not found in current directory.
    pause
    exit /b 1
)

:: Run the application
echo.
echo Starting Fake News Detection System in %RUN_MODE% mode...
echo Command: %PYTHON% %APP_FILE% %ARGS%
%PYTHON% %APP_FILE% %ARGS%
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to run %APP_FILE%. Check error messages above.
    pause
    exit /b 1
)

:: Keep window open for GUI mode or errors
echo.
echo Program execution completed.
if "!RUN_MODE!"=="gui" (
    echo GUI mode complete. Press any key to exit.
    pause >nul
) else if "!RUN_MODE!"=="api" (
    echo API server has stopped. Press any key to exit.
    pause >nul
) else (
    pause
)

:: Deactivate virtual environment
echo Deactivating virtual environment...
deactivate

exit /b 0
```