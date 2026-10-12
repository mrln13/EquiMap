@echo off
REM EquiMap launcher. Double-click (or use the Desktop shortcut) to start.
REM No dataset path is needed here - EquiMap opens a folder picker on launch.
setlocal
set "CONDA_ROOT=%USERPROFILE%\miniforge3"
REM viewer.py lives in the repo root, one level up from this installer folder.
cd /d "%~dp0.."

if not exist "%CONDA_ROOT%\Scripts\activate.bat" (
  echo Could not find the EquiMap environment. Please run install.bat first.
  pause
  exit /b 1
)

call "%CONDA_ROOT%\Scripts\activate.bat" equimap
REM --preview-auto + --elements auto = sensible defaults; --dir is intentionally
REM omitted so the viewer prompts for the dataset folder.
python viewer.py --preview-auto --elements auto
