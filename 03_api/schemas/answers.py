"""
답변 관련 스키마
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Citation(BaseModel):
    """참조 문서 정보"""
    chunk_id: str = Field(..., description="청크 고유 ID")
    doc_title: str = Field(..., description="문서 제목 (예: 식품위생법 제48조)")
    score: float = Field(..., description="관련도 점수 (0~1)")
    text: str = Field(..., description="청크 전체 텍스트 (매뉴얼의 경우 테이블 포함 평탄화됨)")


class AnswerRequest(BaseModel):
    """답변 생성 요청"""
    query_id: str = Field(..., description="질문 ID (QueryResponse에서 받은 값)")
    selected_categories: List[str] = Field(..., description="선택한 카테고리 ID 목록 (예: ['LAW_가축전염병예방법', 'MANUAL_HACCP_관리'])", min_length=1, max_length=3)


class AnswerDetail(BaseModel):
    """답변 상세 정보"""
    text: str = Field(..., description="생성된 답변 텍스트")
    disclaimer: Optional[str] = Field(None, description="면책 사항")


class Timings(BaseModel):
    """처리 시간 정보"""
    retrieval_ms: int = Field(..., description="검색 소요 시간 (밀리초)")
    generation_ms: int = Field(..., description="LLM 생성 소요 시간 (밀리초)")


class AnswerResponse(BaseModel):
    """답변 응답 (동기 - 즉시 반환)"""
    query_id: str = Field(..., description="질문 ID")
    answer_id: str = Field(..., description="답변 고유 ID")
    status: str = Field(default="succeeded", description="상태: 'succeeded'")
    answer: AnswerDetail = Field(..., description="답변 상세 정보")
    citations: List[Citation] = Field(..., description="참조한 문서 목록")
    timings: Timings = Field(..., description="처리 시간 정보")
    created_at: str = Field(..., description="생성 시작 시각 (ISO 8601)")
    completed_at: str = Field(..., description="생성 완료 시각 (ISO 8601)")


class AnswerAsyncResponse(BaseModel):
    """답변 요청 응답 (비동기 - 접수 확인)"""
    answer_id: str = Field(..., description="답변 고유 ID")
    status: str = Field(default="processing", description="상태: 'processing'")
    estimated_seconds: int = Field(default=10, description="예상 소요 시간 (초)")


class AnswerStatusResponse(BaseModel):
    """답변 상태 조회 응답 (비동기 polling용)"""
    answer_id: str = Field(..., description="답변 고유 ID")
    status: str = Field(..., description="상태: 'processing' | 'completed' | 'failed'")
    answer: str | None = Field(None, description="생성된 답변 (완료 시)")
    disclaimer: Optional[str] = Field(None, description="면책 사항")
    citations: List[Citation] | None = Field(None, description="참조 문서 (완료 시)")
    created_at: str | None = Field(None, description="완료 시각 (ISO 8601)")
    error_message: str | None = Field(None, description="에러 메시지 (실패 시)")

