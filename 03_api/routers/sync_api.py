"""
동기 API 엔드포인트

4단계 API 구현 (엑셀 요구사항) - 동기 방식 전용
"""
from fastapi import APIRouter, HTTPException
from schemas.queries import QueryRequest, QueryResponse
from schemas.answers import AnswerRequest, AnswerResponse
from schemas.feedback import ChunkFeedbackRequest
from schemas.chunks import ChunkDetailResponse
from services import query_service, answer_service, feedback_service, chunk_service

router = APIRouter()


@router.post("/queries", response_model=QueryResponse, summary="1. 질문 → 카테고리 추천")
def create_query(req: QueryRequest):
    """
    사용자 질문을 받아 관련 카테고리를 추천합니다.
    
    - **question**: 사용자 질문 (필수)
    - **scope**: 검색 범위 ('law', 'manual', 'all', 기본값: 'all')
    
    **응답**: 추천 카테고리 목록 (최대 5개)
    """
    return query_service.search_categories(req)


@router.post("/answers", response_model=AnswerResponse, summary="2. 답변 생성")
def create_answer(req: AnswerRequest):
    """
    선택한 카테고리를 기반으로 답변을 생성합니다.
    
    **동작:**
    - 5~10초 대기 후 답변 즉시 반환 (동기 방식)
    
    **요청:**
    - **query_id**: 이전에 받은 질문 ID (POST /queries에서 받은 값)
    - **selected_categories**: 선택한 카테고리 목록 (최대 5개)
    
    **응답:**
    - 생성된 답변 + 참조 문서
    """
    return answer_service.generate_sync(req)


@router.post("/feedback/chunks", status_code=204, summary="3. 청크 평가 저장")
def save_feedback(req: ChunkFeedbackRequest):
    """
    여러 청크에 대한 사용자 평가를 한 번에 저장합니다.
    
    - **answer_id**: 답변 ID
    - **query_id**: 질문 ID
    - **feedback**: 청크별 평가 목록 [{"chunk_id": "...", "feedback": "positive"}, ...]
    - **meta**: (선택) 사용자 정보 {"user_id": "...", "session_id": "..."}
    
    **응답**: 204 No Content (피드백 저장 완료, 응답 본문 없음)
    """
    feedback_service.save_chunk_feedback(req)
    return None


@router.get("/answers/{answer_id}/chunks/{chunk_id}", response_model=ChunkDetailResponse, summary="4. 청크 상세 조회")
def get_chunk_detail(answer_id: str, chunk_id: str):
    """
    특정 청크의 전체 텍스트를 조회합니다.
    
    - **answer_id**: 답변 ID
    - **chunk_id**: 청크 ID
    
    **응답**: 청크 전체 텍스트
    """
    return chunk_service.get_chunk_detail(answer_id, chunk_id)

