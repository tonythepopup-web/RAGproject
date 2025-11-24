#!/bin/bash
# API 서버 실행 스크립트 (Linux/Mac)

echo "🚀 HACCP RAG API 서버 시작..."
echo "📍 Swagger UI: http://localhost:8000/docs"
echo "📍 ReDoc: http://localhost:8000/redoc"
echo ""

# 03_api 폴더로 이동
cd "$(dirname "$0")"

# Uvicorn으로 서버 시작
uvicorn main:app --reload --host 0.0.0.0 --port 8000

