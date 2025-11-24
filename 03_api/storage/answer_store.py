"""
답변 저장소 (메모리 기반)

프로덕션에서는 DB(PostgreSQL, MongoDB) 또는 Redis 사용 권장
"""
from typing import Dict, Any, Optional
from datetime import datetime

# 메모리 기반 저장소
_answers: Dict[str, Dict[str, Any]] = {}


def save_answer(
    answer_id: str,
    query_id: str,
    answer: str,
    citations: list,
    status: str = "completed",
    confidence: Optional[float] = None,
    disclaimer: Optional[str] = None
) -> None:
    """답변 저장"""
    _answers[answer_id] = {
        "answer_id": answer_id,
        "query_id": query_id,
        "answer": answer,
        "citations": citations,
        "status": status,
        "confidence": confidence,
        "disclaimer": disclaimer,
        "created_at": datetime.now().isoformat()
    }


def get_answer(answer_id: str) -> Optional[Dict[str, Any]]:
    """답변 조회"""
    return _answers.get(answer_id)


def list_answers(limit: int = 100) -> list:
    """전체 답변 목록 (최근 N개)"""
    answers = list(_answers.values())
    answers.sort(key=lambda x: x["created_at"], reverse=True)
    return answers[:limit]


def get_answers_by_query(query_id: str) -> list:
    """특정 질문에 대한 모든 답변"""
    return [
        a for a in _answers.values()
        if a["query_id"] == query_id
    ]

