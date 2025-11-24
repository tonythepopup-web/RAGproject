"""
청크 상세 조회 서비스
"""
from fastapi import HTTPException
from schemas.chunks import ChunkDetailResponse
from storage import chunk_store


def get_chunk_detail(answer_id: str, chunk_id: str, query_id: str = None) -> ChunkDetailResponse:
    """
    청크의 상세 정보를 조회합니다.
    
    Args:
        answer_id: 답변 ID
        chunk_id: 청크 ID
        query_id: 질문 ID (선택)
    
    Returns:
        ChunkDetailResponse
    """
    chunk = chunk_store.get_chunk_detail(answer_id, chunk_id)
    
    if not chunk:
        raise HTTPException(
            status_code=404,
            detail=f"Chunk ID '{chunk_id}' not found in Answer ID '{answer_id}'"
        )
    
    # query_id 추론 (필요시)
    if not query_id:
        from storage import answer_store
        answer = answer_store.get_answer(answer_id)
        query_id = answer.get("query_id") if answer else "unknown"
    
    return ChunkDetailResponse(
        answer_id=answer_id,
        query_id=query_id,
        chunk_id=chunk_id,
        chunk_text=chunk.get("text", "")
    )

