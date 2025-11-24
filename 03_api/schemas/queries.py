"""
질문 관련 스키마
"""
from pydantic import BaseModel, Field
from typing import List


class Category(BaseModel):
    """카테고리 정보"""
    category_id: str = Field(..., description="카테고리 고유 ID (LAW_제목 또는 MANUAL_제목)")
    label: str = Field(..., description="카테고리 표시 이름 (순수 제목만)")
    score: float = Field(..., description="관련도 점수 (0~1)")


class QueryRequest(BaseModel):
    """질문 요청"""
    question: str = Field(..., description="사용자 질문", min_length=1)
    scope: str = Field(default="all", description="검색 범위: 'law', 'manual', 'all'")


class QueryResponse(BaseModel):
    """질문 응답 (카테고리 추천)"""
    query_id: str = Field(..., description="질문 고유 ID")
    scope: str = Field(..., description="검색 범위 ('law', 'manual', 'all')")
    question: str = Field(..., description="사용자 질문")
    category_candidates: List[Category] = Field(..., description="추천 카테고리 목록 (전체 1개 + 추천 최대 2개 = 총 3개)")
    created_at: str = Field(..., description="생성 시각 (ISO 8601)")

