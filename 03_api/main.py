"""
HACCP RAG API 서버

4단계 API 구현 (엑셀 요구사항)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import sync_api

app = FastAPI(
    title="HACCP RAG API",
    description="법률/매뉴얼 질의응답 API",
    version="1.0.0"
)

# CORS 설정 (모든 출처 허용 - 필요시 수정)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 4단계 API (동기 방식 전용)
app.include_router(
    sync_api.router,
    tags=["API"]
)


@app.get("/", tags=["Root"])
def root():
    """API 루트 - 사용 가능한 엔드포인트 안내"""
    return {
        "message": "HACCP RAG API (4단계)",
        "version": "1.0.0",
        "endpoints": {
            "1. queries": "POST /queries",
            "2. answers": "POST /answers",
            "3. feedback": "POST /feedback/chunks",
            "4. chunk_detail": "GET /answers/{answer_id}/chunks/{chunk_id}"
        },
        "docs": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health", tags=["Health"])
def health_check():
    """서버 상태 확인"""
    return {"status": "healthy"}

