@echo off
rem Reachy Cafe - Windows 시작 스크립트. 더블클릭하면 메뉴가 뜬다 (터미널: start_windows.bat [명령] [인자...]).
rem   setup preflight spike calibrate kiosk mock sim test  (start_linux.sh, start_mac.command와 같은 명령)
rem GStreamer는 pip 패키지(gstreamer-bundle)에 들어 있다. NVIDIA GPU가 있으면 GPU 음성 인식(cuda)을 함께 설치한다.
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if exist "%USERPROFILE%\.local\bin\uv.exe" set "PATH=%USERPROFILE%\.local\bin;%PATH%"
if exist "%USERPROFILE%\.cargo\bin\uv.exe" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

if "%~1"=="" goto :menu
set "CMD=%~1"
set "ARGS="
shift
:collect
if "%~1"=="" goto :run
set ARGS=%ARGS% %1
shift
goto :collect

:run
call :dispatch %CMD%
exit /b %errorlevel%

:menu
echo Reachy Cafe (Windows)
echo   1) 설치 (처음 한 번)
echo   2) 사전 점검 (로봇 안 움직임)
echo   3) 연결 점검 (로봇 움직임)
echo   4) 배치 보정 (화면·슬롯 위치, 고개 방향)
echo   5) 키오스크 실행 (실제 로봇)
echo   6) 키오스크 실행 (로봇 없이 mock)
echo   7) MuJoCo 시뮬 점검
echo   8) 테스트
set "N="
set /p N=번호:
set "ARGS="
if "%N%"=="1" call :dispatch setup
if "%N%"=="2" call :dispatch preflight
if "%N%"=="3" call :dispatch spike
if "%N%"=="4" call :dispatch calibrate
if "%N%"=="5" call :dispatch kiosk
if "%N%"=="6" call :dispatch mock
if "%N%"=="7" call :dispatch sim
if "%N%"=="8" call :dispatch test
echo.
pause
exit /b 0

:dispatch
if /i "%~1"=="setup" goto :setup
if /i "%~1"=="preflight" (call :py tools\preflight.py --load-models %ARGS% & exit /b)
if /i "%~1"=="spike" (call :py tools\spike_connection.py %ARGS% & exit /b)
if /i "%~1"=="calibrate" (call :py tools\spike_connection.py --skip audio --skip asr --skip motion --aim --calibrate %ARGS% & exit /b)
if /i "%~1"=="kiosk" goto :kiosk
if /i "%~1"=="mock" (call :py -m reachy_cafe --config config/dev_laptop.toml %ARGS% & exit /b)
if /i "%~1"=="sim" (call :py tools\sim_check.py %ARGS% & exit /b)
if /i "%~1"=="test" (call :py -m pytest -q %ARGS% & exit /b)
if /i "%~1"=="bundle" (call :py tools\make_bundle.py %ARGS% & exit /b)
echo [reachy-cafe] 모르는 명령: %~1 (setup preflight spike calibrate kiosk mock sim test)
exit /b 2

:kiosk
if exist config\calibration.toml (
  call :py -m reachy_cafe --config config/calibration.toml %ARGS%
) else (
  call :py -m reachy_cafe %ARGS%
)
exit /b

:py
if not exist "%PY%" (
  echo [reachy-cafe] .venv가 없습니다. 먼저: start_windows.bat setup
  exit /b 1
)
"%PY%" -X utf8 %*
exit /b

:setup
where uv >nul 2>nul
if errorlevel 1 (
  echo [reachy-cafe] uv가 없습니다. PowerShell에서 아래 한 줄로 설치하고, 새 창에서 다시 실행하세요:
  echo     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  exit /b 1
)
set "EXTRAS=dev,robot,speech,sim"
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  set "EXTRAS=dev,robot,speech,sim,cuda"
  echo [reachy-cafe] NVIDIA GPU 발견 - GPU 음성 인식^(cuda^) 포함
) else (
  echo [reachy-cafe] NVIDIA GPU 없음 - 음성 인식은 CPU에서 ^(자동 전환^)
)
if not exist "%PY%" (
  uv venv --python 3.12 .venv
  if errorlevel 1 (
    echo [reachy-cafe] 실패: .venv를 만들지 못했습니다
    exit /b 1
  )
)
uv pip install --python "%PY%" -e ".[%EXTRAS%]"
if errorlevel 1 (
  echo [reachy-cafe] 실패: 위 메시지를 보세요
  exit /b 1
)
echo [reachy-cafe] 설치 끝. 다음: start_windows.bat preflight
exit /b 0
