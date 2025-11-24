#!/bin/bash
# bge-m3 임베딩 서버 시작 스크립트 (TEI 사용)
# Text Embeddings Inference: https://github.com/huggingface/text-embeddings-inference

echo "======================================"
echo "  bge-m3 임베딩 서버 시작 (TEI)"
echo "======================================"
echo

# Docker로 TEI 실행
docker run -d \
    --name tei-bge-m3 \
    --gpus all \
    -p 8401:80 \
    -v $PWD/models:/data \
    ghcr.io/huggingface/text-embeddings-inference:latest \
    --model-id BAAI/bge-m3 \
    --revision main

echo
echo "✅ 임베딩 서버 시작됨"
echo "   엔드포인트: http://localhost:8401"
echo "   모델: BAAI/bge-m3"
echo
echo "테스트:"
echo '  curl http://localhost:8401/embed \\'
echo '    -X POST \\'
echo '    -d ''{"inputs":"HACCP 인증 기준은?"}'' \\'
echo '    -H ''Content-Type: application/json'''
echo
echo "중지: docker stop tei-bge-m3"
echo "제거: docker rm tei-bge-m3"

