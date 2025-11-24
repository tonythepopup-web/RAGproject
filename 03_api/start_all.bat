@echo off
REM Windows용 서버 자동 기동 스크립트
REM 사용법: start_all.bat

echo ====================================
echo   HACCP RAG 서버 자동 기동
echo ====================================
echo.

REM 가상환경 활성화 (있는 경우)
if exist .venv\Scripts\activate.bat (
    echo 가상환경 활성화 중...
    call .venv\Scripts\activate.bat
)

REM Python 스크립트 실행
python start_services.py

pause

