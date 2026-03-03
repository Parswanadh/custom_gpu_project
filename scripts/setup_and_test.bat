@echo off
REM ============================================================================
REM setup_and_test.bat - Install pwsh via conda + run all GPU tests
REM
REM HOW TO USE:
REM   1. Open Anaconda Prompt (or any terminal with conda in PATH)
REM   2. cd D:\projects\bitbybit\custom_gpu_project
REM   3. scripts\setup_and_test.bat
REM
REM Or just double-click this file from Explorer.
REM ============================================================================
setlocal

echo.
echo ================================================================
echo  BitbyBit GPU - Setup and Test
echo ================================================================
echo.

REM Step 0: Try to activate conda if not already active
if "%CONDA_DEFAULT_ENV%"=="" (
    echo [INFO] Conda env not active. Trying to initialize...
    REM Common conda install locations
    if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
        call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
    ) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
        call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%USERPROFILE%\miniconda3"
    ) else if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
        call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3"
    ) else if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" (
        call "C:\ProgramData\miniconda3\Scripts\activate.bat" "C:\ProgramData\miniconda3"
    )
)

REM Step 1: Check if pwsh is already available
where pwsh >nul 2>&1
if %errorlevel% EQU 0 (
    echo [OK] pwsh already installed
    goto :run_tests
)

REM Step 2: Install pwsh via conda
echo [INFO] Installing PowerShell 7 via conda-forge...
where conda >nul 2>&1
if %errorlevel% NEQ 0 (
    echo [ERROR] conda not found in PATH.
    echo Please open Anaconda Prompt and run this script from there.
    echo Or manually run: conda install -y -c conda-forge powershell
    pause
    exit /b 1
)

conda install -y -c conda-forge powershell
if %errorlevel% NEQ 0 (
    echo [WARN] conda install failed, trying winget...
    where winget >nul 2>&1
    if %errorlevel% EQU 0 (
        winget install --id Microsoft.PowerShell --source winget --accept-package-agreements --accept-source-agreements
    ) else (
        echo [ERROR] Could not install pwsh. Please install manually:
        echo   conda install -c conda-forge powershell
        echo   OR: winget install Microsoft.PowerShell
        pause
        exit /b 1
    )
)

echo.
echo [OK] PowerShell installed. Verifying...
where pwsh >nul 2>&1
if %errorlevel% NEQ 0 (
    echo [WARN] pwsh not in PATH yet. You may need to restart your terminal.
)
echo.

:run_tests
echo ================================================================
echo  Running GPU Test Suite (24 modules, 7 phases)
echo ================================================================
echo.

REM Run the Python test runner (more robust output parsing)
where python >nul 2>&1
if %errorlevel% EQU 0 (
    python "%~dp0run_tests.py"
    goto :done
)

REM Fallback to batch runner
echo [INFO] Python not found, using batch runner...
call "%~dp0run_tests.bat"

:done
echo.
echo Test logs saved to: sim\*_output.log
echo.
pause
