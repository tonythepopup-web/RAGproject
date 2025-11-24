"""
피드백 서비스
"""
from fastapi import HTTPException
from schemas.feedback import ChunkFeedbackRequest
from storage import feedback_store, answer_store, query_store


def save_chunk_feedback(req: ChunkFeedbackRequest) -> None:
    """
    여러 청크에 대한 피드백을 한 번에 저장합니다.
    
    Args:
        req: ChunkFeedbackRequest (answer_id, query_id, feedback[], meta)
    
    Returns:
        None (204 No Content)
    """
    # answer_id 존재 확인
    answer = answer_store.get_answer(req.answer_id)
    if not answer:
        raise HTTPException(
            status_code=404,
            detail=f"Answer ID '{req.answer_id}'를 찾을 수 없습니다."
        )
    
    # query_id 존재 확인
    query = query_store.get_query(req.query_id)
    if not query:
        raise HTTPException(
            status_code=404,
            detail=f"Query ID '{req.query_id}'를 찾을 수 없습니다."
        )
    
    question = query.get("question", "")
    citations = answer.get("citations", [])
    
    # meta 정보 추출
    user_id = req.meta.user_id if req.meta else None
    session_id = req.meta.session_id if req.meta else None
    
    # 각 피드백 처리
    for fb in req.feedback:
        # 프론트엔드의 positive/negative를 점수로 변환
        score = fb.score
        fb_type = "positive" if score >= 3.0 else "negative"
        
        # 피드백 저장
        feedback_store.save_feedback(
            answer_id=req.answer_id,
            query_id=req.query_id,
            chunk_id=fb.chunk_id,
            score=score,
            user_id=user_id,
            session_id=session_id
        )
        
        # ===== Triplet 자동 로깅 =====
        # positive: 긍정 샘플, negative: 부정 샘플
        chunk_data = None
        for c in citations:
            if c.get("chunk_id") == fb.chunk_id:
                chunk_data = c
                break
        
        if chunk_data:
            text = chunk_data.get("text", "")
            doc_title = chunk_data.get("doc_title", "")
            
            if fb_type == "positive":
                # Positive 피드백
                feedback_store.save_triplet_log(
                    query=question,
                    positives=[text] if text else [],
                    negatives=[],
                    pos_sources=[doc_title] if doc_title else None,
                    neg_sources=None,
                    extra_meta={
                        "answer_id": req.answer_id,
                        "query_id": req.query_id,
                        "chunk_id": fb.chunk_id,
                        "feedback": fb_type,
                        "score": score
                    }
                )
            else:
                # Negative 피드백
                feedback_store.save_triplet_log(
                    query=question,
                    positives=[],
                    negatives=[text] if text else [],
                    pos_sources=None,
                    neg_sources=[doc_title] if doc_title else None,
                    extra_meta={
                        "answer_id": req.answer_id,
                        "query_id": req.query_id,
                        "chunk_id": fb.chunk_id,
                        "feedback": fb_type,
                        "score": score
                    }
                )
    
    # 204 No Content (return None)

