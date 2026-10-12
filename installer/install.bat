@echo off
setlocal EnableExtensions EnableDelayedExpansion
title EquiMap installer

REM ===========================================================================
REM  EquiMap installer for Windows.
REM
REM  This file lives in the "installer" subfolder; viewer.py and the rest of the
REM  program are in the parent folder. Paths below account for that.
REM
REM  What it does, with no admin rights required:
REM    1. Installs a private Miniforge (conda) into your user profile if needed.
REM    2. Creates the "equimap" environment from environment.yml.
REM    3. Downloads the Julia + MAGEMin runtime so modelling works immediately.
REM    4. Puts an "EquiMap" shortcut on your Desktop.
REM
REM  Just double-click this file. It needs an internet connection and ~10-15
REM  minutes the first time (most of that is the Julia/MAGEMin download).
REM ===========================================================================

set "INSTALLER_DIR=%~dp0"
set "LOG=%INSTALLER_DIR%install_log.txt"

REM --- Mirror all output to a log file (tee) for troubleshooting --------------
REM On first launch, re-run this script with combined stdout+stderr piped through
REM PowerShell's Tee-Object, so the user sees progress live AND a full transcript
REM is written next to this script. _EQUIMAP_LOGGING (inherited by the relaunched
REM child) prevents infinite relaunching. The log is overwritten each run.
if not defined _EQUIMAP_LOGGING (
  set "_EQUIMAP_LOGGING=1"
  echo Saving an installation log to:
  echo   %LOG%
  echo(
  "%~f0" %* 2>&1 | powershell -NoProfile -ExecutionPolicy Bypass -Command "$input | Tee-Object -FilePath '%LOG%'"
  echo(
  echo ===========================================================
  echo   A full log was saved to:
  echo     %LOG%
  echo   Please include it when reporting an installation problem.
  echo ===========================================================
  pause
  exit /b
)

set "CONDA_ROOT=%USERPROFILE%\miniforge3"
set "ENV_NAME=equimap"
set "CONDA_EXE=%CONDA_ROOT%\Scripts\conda.exe"
set "MINIFORGE_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
set "MINIFORGE_EXE=%TEMP%\Miniforge3-Windows-x86_64.exe"

echo(
echo ===========================================================
echo   EquiMap installer
echo   Target conda : %CONDA_ROOT%
echo   Environment  : %ENV_NAME%
echo   Started      : %DATE% %TIME%
echo ===========================================================
echo(

REM --- Guard: a space in the install path breaks conda. -----------------------
REM Detect a space via substring substitution.
if not "%CONDA_ROOT%"=="%CONDA_ROOT: =%" (
  echo [ERROR] The install path contains a space:
  echo         %CONDA_ROOT%
  echo         conda cannot install reliably there. Edit CONDA_ROOT near the top
  echo         of this script to a space-free path you can write to
  echo         ^(e.g. C:\equimap-conda^), then run it again.
  goto :fail
)

REM --- Windows on ARM: conda-forge has no win-arm64 builds for this stack
REM     (numba/imagecodecs/napari deps, etc.), so target win-64 and let the
REM     OS run it under x64 emulation. Harmless on real x64 machines. ----------
set "_ARCH=%PROCESSOR_ARCHITECTURE%"
if /I "%PROCESSOR_ARCHITEW6432%"=="ARM64" set "_ARCH=ARM64"
if /I "%_ARCH%"=="ARM64" (
  echo Detected Windows on ARM - building an x64 environment ^(runs under emulation^).
  echo It will work but may be somewhat slower than on a native x64 PC.
  set "CONDA_SUBDIR=win-64"
)

REM --- Step 1: ensure Miniforge ----------------------------------------------
if exist "%CONDA_EXE%" (
  echo [1/4] Miniforge already present at %CONDA_ROOT% - reusing it.
) else (
  echo [1/4] Downloading Miniforge installer...
  call :download "%MINIFORGE_URL%" "%MINIFORGE_EXE%"
  if errorlevel 1 (
    echo [ERROR] Could not download Miniforge. Check your internet connection.
    goto :fail
  )
  echo       Installing Miniforge ^(silent, no admin^)...
  REM NSIS: /D= must be the LAST argument and must NOT be quoted.
  "%MINIFORGE_EXE%" /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=%CONDA_ROOT%
  if not exist "%CONDA_EXE%" (
    echo [ERROR] Miniforge installation did not complete.
    goto :fail
  )
  del /q "%MINIFORGE_EXE%" 2>nul
)

REM --- Step 2: create or update the environment ------------------------------
if exist "%CONDA_ROOT%\envs\%ENV_NAME%" (
  echo [2/4] Environment "%ENV_NAME%" exists - updating from environment.yml...
  echo       ^(several minutes; progress is summarised, not animated^)
  "%CONDA_EXE%" env update -q -n %ENV_NAME% -f "%INSTALLER_DIR%environment.yml" --prune
) else (
  echo [2/4] Creating environment "%ENV_NAME%" from environment.yml...
  echo       ^(this is the long step - several minutes^)
  "%CONDA_EXE%" env create -q -f "%INSTALLER_DIR%environment.yml"
)
if errorlevel 1 (
  echo [ERROR] Conda could not build the environment. See the messages above.
  goto :fail
)

REM On ARM, record the env's subdir so any later conda op on it stays win-64.
if /I "%_ARCH%"=="ARM64" >"%CONDA_ROOT%\envs\%ENV_NAME%\.condarc" echo subdir: win-64

REM --- Step 3: install the Julia + MAGEMin runtime ---------------------------
echo [3/4] Installing the Julia + MAGEMin runtime for modelling...
echo       ^(first time only; downloads Julia ~1.11 + MAGEMin_C.jl^)
"%CONDA_EXE%" run --no-capture-output -n %ENV_NAME% python -c "import petthermotools as ptt; ptt.install_MAGEMinCalc()"
if errorlevel 1 (
  echo [WARN] The MAGEMin/Julia setup did not finish cleanly. EquiMap will still
  echo        start and you can view/segment data; the MAGEMin backend may need
  echo        a retry ^(re-run this installer^) before modelling works.
)

REM --- Step 4: desktop shortcut ----------------------------------------------
REM Target the launcher in this installer folder; working dir = repo root (parent).
echo [4/4] Creating the Desktop shortcut...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\EquiMap.lnk'); $s.TargetPath='%INSTALLER_DIR%EquiMap.bat'; $s.WorkingDirectory=(Resolve-Path '%INSTALLER_DIR%..').Path; $s.IconLocation='%SystemRoot%\System32\shell32.dll,13'; $s.Description='Launch EquiMap'; $s.Save()"

echo(
echo ===========================================================
echo   Done. Double-click "EquiMap" on your Desktop to start.
echo   It will ask you to pick a dataset folder.
echo ===========================================================
echo(
exit /b 0

REM --- helper: download(url, outfile) ----------------------------------------
:download
where curl >nul 2>nul
if not errorlevel 1 (
  curl -sS -L --fail --retry 3 -o "%~2" "%~1"
  exit /b %errorlevel%
)
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { Invoke-WebRequest -Uri '%~1' -OutFile '%~2' -UseBasicParsing; exit 0 } catch { exit 1 }"
exit /b %errorlevel%

:fail
echo(
echo Installation failed. Nothing was changed system-wide; you can safely
echo re-run this installer after fixing the issue above.
echo(
exit /b 1
