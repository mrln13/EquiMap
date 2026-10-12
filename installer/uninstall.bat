@echo off
setlocal EnableExtensions
title EquiMap uninstaller

REM ===========================================================================
REM  EquiMap uninstaller for Windows. Removes everything install.bat created:
REM    - the EquiMap Desktop shortcut
REM    - the private conda install + equimap environment (CONDA_ROOT below)
REM    - (after asking) the shared Julia + MAGEMin depot
REM  It does NOT touch PATH, the registry, or your shell profile (install.bat
REM  never changed those), and it does NOT delete this EquiMap program folder.
REM
REM  If you changed CONDA_ROOT in install.bat, change it to match here too.
REM ===========================================================================

set "CONDA_ROOT=%USERPROFILE%\miniforge3"
set "JULIA_DEPOT=%USERPROFILE%\.julia"

echo(
echo ===========================================================
echo   EquiMap uninstaller
echo ===========================================================
echo This will remove:
echo   - the EquiMap Desktop shortcut
echo   - the conda install + equimap environment:
echo       %CONDA_ROOT%
echo It will ASK before removing the shared Julia/MAGEMin folder:
echo       %JULIA_DEPOT%
echo It will NOT delete this EquiMap program folder (delete it yourself after).
echo(
set /p "GO=Proceed? [y/N] "
if /I not "%GO%"=="y" (
  echo Cancelled. Nothing was removed.
  pause
  exit /b 0
)

REM --- 1. Desktop shortcut (use the real, possibly OneDrive-redirected, Desktop)
echo(
echo [1/3] Removing Desktop shortcut...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p=[Environment]::GetFolderPath('Desktop')+'\EquiMap.lnk'; if (Test-Path $p) { Remove-Item $p -Force; Write-Host ('      removed ' + $p) } else { Write-Host '      (no shortcut found)' }"

REM --- 2. conda + environment
echo(
echo [2/3] Removing conda + the equimap environment...
if exist "%CONDA_ROOT%" (
  rmdir /s /q "%CONDA_ROOT%"
  if exist "%CONDA_ROOT%" (
    echo       [WARN] Could not fully remove %CONDA_ROOT%.
    echo              Close any open EquiMap / conda windows and run this again.
  ) else (
    echo       removed %CONDA_ROOT%
  )
) else (
  echo       (not found at %CONDA_ROOT% - skipping)
)

REM --- 3. Julia depot (ask first - it is the SHARED Julia depot)
echo(
echo [3/3] The Julia + MAGEMin folder is the SHARED Julia depot:
echo       %JULIA_DEPOT%
echo       Remove it ONLY if you do not use Julia for anything else.
set /p "DJ=Remove it now? [y/N] "
if /I "%DJ%"=="y" (
  if exist "%JULIA_DEPOT%" (
    rmdir /s /q "%JULIA_DEPOT%"
    if exist "%JULIA_DEPOT%" (
      echo       [WARN] Could not fully remove %JULIA_DEPOT%.
    ) else (
      echo       removed %JULIA_DEPOT%
    )
  ) else (
    echo       (not found - skipping)
  )
) else (
  echo       kept %JULIA_DEPOT%
)

echo(
echo ===========================================================
echo   Done. You can now delete this EquiMap program folder.
echo ===========================================================
pause
exit /b 0
