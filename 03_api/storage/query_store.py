"""
질문 저장소 (메모리 기반)

프로덕션에서는 DB(PostgreSQL, MongoDB) 또는 Redis 사용 권장
"""
from typing import Dict, Any, Optional
from datetime import datetime

# 메모리 기반 저장소
_queries: Dict[str, Dict[str, Any]] = {}


def save_query(
    query_id: str,
    question: str,
    scope: str,
    recommended_categories: list
) -> None:
    """질문 저장"""
    _queries[query_id] = {
        "query_id": query_id,
        "question": question,
        "scope": scope,
        "recommended_categories": recommended_categories,
        "created_at": datetime.now().isoformat()
    }


def get_query(query_id: str) -> Optional[Dict[str, Any]]:
    """질문 조회"""
    return _queries.get(query_id)


def list_queries(limit: int = 100) -> list:
    """전체 질문 목록 (최근 N개)"""
    queries = list(_queries.values())
    queries.sort(key=lambda x: x["created_at"], reverse=True)
    return queries[:limit]

