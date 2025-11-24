"""
청크 상세 정보 저장소

실제로는 citations 정보에 chunk_id와 text가 있으므로
answer_store의 citations를 활용
"""
from typing import Optional, Dict, Any
from storage import answer_store


def get_chunk_detail(answer_id: str, chunk_id: str) -> Optional[Dict[str, Any]]:
    """
    특정 청크의 상세 정보 조회
    
    Args:
        answer_id: 답변 ID
        chunk_id: 청크 ID
    
    Returns:
        {"chunk_id": ..., "text": ..., "source": ...} or None
    """
    answer = answer_store.get_answer(answer_id)
    
    if not answer:
        return None
    
    # citations에서 해당 chunk_id 찾기
    citations = answer.get("citations", [])
    
    for citation in citations:
        if citation.get("chunk_id") == chunk_id:
            return {
                "chunk_id": citation.get("chunk_id"),
                "text": citation.get("text", ""),
                "source": citation.get("source", ""),
                "category": citation.get("category", "")
            }
    
    return None

