@echo off
REM Windows용 프론트엔드 실행 스크립트

echo ========================================
echo   법률 검색 플랫폼 - 프론트엔드 실행
echo ========================================
echo.

REM 백엔드 서버 체크
echo [1/2] 백엔드 서버 연결 확인 중...
curl -s http://localhost:8000/health > nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [경고] 백엔드 서버가 실행되지 않았습니다!
    echo.
    echo 먼저 백엔드 서버를 실행해주세요:
    echo   cd 03_api
    echo   python start_services.py
    echo.
    pause
    exit /b 1
)

echo OK - 백엔드 서버 정상 동작 중
echo.

REM Python HTTP 서버 실행
echo [2/2] 프론트엔드 서버 시작 중...
echo.
echo 브라우저에서 다음 주소로 접속하세요:
echo   http://localhost:3000
echo.
echo 종료하려면 Ctrl+C를 누르세요
echo.
echo ========================================
echo.

python -m http.server 3000

