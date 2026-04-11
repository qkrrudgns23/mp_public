@echo off
setlocal
cd /d "%~dp0"
set "BLEND=cybertruck (final).blend"
set "PY=export_cybertruck_glb.py"
if not exist "%BLEND%" (
  echo Missing "%BLEND%" in this folder.
  exit /b 1
)
for %%B in (
  "C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"
  "C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"
  "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
  "C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"
  "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe"
  "C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"
) do if exist %%B (
  echo Using %%B
  %%B "%BLEND%" --background --python "%PY%"
  exit /b %ERRORLEVEL%
)
where blender >nul 2>nul
if %ERRORLEVEL% equ 0 (
  blender "%BLEND%" --background --python "%PY%"
  exit /b %ERRORLEVEL%
)
echo Blender not found. Install from https://www.blender.org/download/ then run:
echo   blender "%BLEND%" --background --python "%PY%"
exit /b 1
