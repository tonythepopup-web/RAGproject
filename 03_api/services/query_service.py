"""
질문 처리 서비스

카테고리 추천 로직
"""
import uuid
from typing import Dict, Any
from schemas.queries import QueryRequest, QueryResponse
from adapters.rag_adapter import RAGAdapter
from storage import query_store

# Boot Once: 모듈 로드 시 1회만 초기화
_rag_instance = None


def get_rag_instance() -> RAGAdapter:
    """RAG 인스턴스 지연 로딩 (앱 전체에서 1개만 사용)"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGAdapter()
    return _rag_instance


def search_categories(req: QueryRequest) -> QueryResponse:
    """
    질문 기반 카테고리 추천 (최대 2개)
    
    Args:
        req: QueryRequest (question, scope)
    
    Returns:
        QueryResponse (query_id, scope, question, category_candidates, created_at)
    """
    try:
        from datetime import datetime
        
        # RAG 인스턴스 가져오기
        rag = get_rag_instance()
        
        # 카테고리 추천 (최대 2개)
        categories = rag.get_recommended_categories(
            scope=req.scope,
            question=req.question,
            top_k=2
        )
        
        # 응답 생성
        query_id = str(uuid.uuid4())
        response = QueryResponse(
            query_id=query_id,
            scope=req.scope,
            question=req.question,
            category_candidates=categories,
            created_at=datetime.now().isoformat()
        )
        
        # 저장 (이력 관리)
        query_store.save_query(
            query_id=query_id,
            question=req.question,
            scope=req.scope,
            recommended_categories=categories
        )
        
        return response
    
    except Exception as e:
        from datetime import datetime
        print(f"❌ [search_categories] 오류: {e}")
        # 오류 시 빈 추천 반환
        return QueryResponse(
            query_id=str(uuid.uuid4()),
            scope=req.scope,
            question=req.question,
            category_candidates=[],
            created_at=datetime.now().isoformat()
        )

