#!/bin/bash
# Linux/Mac용 서버 자동 기동 스크립트
# 사용법: bash start_all.sh 또는 ./start_all.sh

echo "===================================="
echo "  HACCP RAG 서버 자동 기동"
echo "===================================="
echo

# 스크립트 디렉토리로 이동
cd "$(dirname "$0")"

# 가상환경 활성화 (있는 경우)
if [ -f ".venv/bin/activate" ]; then
    echo "가상환경 활성화 중..."
    source .venv/bin/activate
fi

# Python 스크립트 실행
python start_services.py

