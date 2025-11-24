#!/bin/bash
# Linux/Mac용 프론트엔드 실행 스크립트

echo "========================================"
echo "  법률 검색 플랫폼 - 프론트엔드 실행"
echo "========================================"
echo ""

# 백엔드 서버 체크
echo "[1/2] 백엔드 서버 연결 확인 중..."
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo ""
    echo "[경고] 백엔드 서버가 실행되지 않았습니다!"
    echo ""
    echo "먼저 백엔드 서버를 실행해주세요:"
    echo "  cd 03_api"
    echo "  python start_services.py"
    echo ""
    exit 1
fi

echo "OK - 백엔드 서버 정상 동작 중"
echo ""

# Python HTTP 서버 실행
echo "[2/2] 프론트엔드 서버 시작 중..."
echo ""
echo "브라우저에서 다음 주소로 접속하세요:"
echo "  http://localhost:3000"
echo ""
echo "종료하려면 Ctrl+C를 누르세요"
echo ""
echo "========================================"
echo ""

python3 -m http.server 3000

