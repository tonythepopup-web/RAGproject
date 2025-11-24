"""
답변 생성 서비스

동기/비동기 답변 생성 로직
"""
import uuid
import threading
from datetime import datetime
from typing import Dict, Any
from fastapi import HTTPException
from schemas.answers import (
    AnswerRequest, 
    AnswerResponse, 
    AnswerAsyncResponse, 
    AnswerStatusResponse
)
from adapters.rag_adapter import RAGAdapter
from services.query_service import get_rag_instance
from storage import query_store, answer_store


# ===== 동기 방식 =====

def generate_sync(req: AnswerRequest) -> AnswerResponse:
    """
    동기 방식 답변 생성 (즉시 완료 대기)
    
    Args:
        req: AnswerRequest (query_id, selected_categories)
    
    Returns:
        AnswerResponse (query_id, answer_id, status, answer, citations, timings, created_at, completed_at)
    """
    try:
        import time
        answer_id = str(uuid.uuid4())
        created_at = datetime.now()
        
        # query_id로부터 질문 복원
        query = query_store.get_query(req.query_id)
        question = query.get("question", "") if query else ""
        
        if not question:
            raise HTTPException(
                status_code=404,
                detail=f"Query ID '{req.query_id}'를 찾을 수 없습니다."
            )
        
        # RAG 인스턴스 가져오기
        rag = get_rag_instance()
        
        # category_id를 RAG 형식으로 변환
        # 예: "LAW_가축전염병예방법" → "법률_가축전염병 예방법" 또는 "MANUAL_전체" → "매뉴얼_전체"
        category_labels = []
        for cat_id in req.selected_categories:
            if cat_id in ["LAW_전체", "MANUAL_전체", "ALL_전체"]:
                # "전체" 옵션 처리 - 타입 prefix 추가
                if cat_id == "LAW_전체":
                    category_labels.append("법률_전체")
                elif cat_id == "MANUAL_전체":
                    category_labels.append("매뉴얼_전체")
                else:  # ALL_전체
                    category_labels.append("전체")
            else:
                # category_id에서 타입과 label 추출
                # "LAW_가축전염병예방법" → "법률_가축전염병 예방법"
                # "MANUAL_HACCP_인증_따라하기" → "매뉴얼_HACCP 인증 따라하기"
                if cat_id.startswith("LAW_"):
                    label = cat_id[4:].replace("_", " ")  # "LAW_" 제거 후 _ → 공백
                    category_labels.append(f"법률_{label}")
                elif cat_id.startswith("MANUAL_"):
                    label = cat_id[7:].replace("_", " ")  # "MANUAL_" 제거 후 _ → 공백
                    category_labels.append(f"매뉴얼_{label}")
                else:
                    # 예외 처리
                    category_labels.append(cat_id.replace("_", " "))
        
        # scope 추론 (카테고리 ID에서)
        scope = _infer_scope(req.selected_categories)
        
        # 답변 생성 (RAG 내부에서 시간 측정)
        result = rag.generate_answer(
            question=question,
            selected_categories=category_labels,  # label 전달
            scope=scope
        )
        
        # RAG에서 측정한 실제 시간 가져오기
        timings = result.get("timings", {"retrieval_ms": 0, "generation_ms": 0})
        retrieval_ms = timings.get("retrieval_ms", 0)
        generation_ms = timings.get("generation_ms", 0)
        
        citations_full = result.get("citations", [])  # text 포함
        
        # API 응답용 citations (text 포함 - 2단계에서 바로 청크 내용 제공)
        from schemas.answers import Citation
        citations_api = [
            Citation(
                chunk_id=c["chunk_id"],
                doc_title=c["doc_title"],
                score=c["score"],
                text=c.get("text", "")  # 청크 전체 텍스트 (매뉴얼은 embedding_text 평탄화)
            )
            for c in citations_full
        ]
        
        # disclaimer 설정
        disclaimer = None
        if scope in ["law", "all"]:
            disclaimer = "본 답변은 법률 자문이 아니며, 참고용으로만 사용하시기 바랍니다."
        
        completed_at = datetime.now()
        
        # 응답 생성 (confidence 제거)
        from schemas.answers import AnswerDetail, Timings
        response = AnswerResponse(
            query_id=req.query_id,
            answer_id=answer_id,
            status="succeeded",
            answer=AnswerDetail(
                text=result["answer"],
                disclaimer=disclaimer
            ),
            citations=citations_api,  # API용 (text 제외)
            timings=Timings(
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms
            ),
            created_at=created_at.isoformat(),
            completed_at=completed_at.isoformat()
        )
        
        # 저장 (이력 관리 - text 포함된 전체 citations 저장)
        answer_store.save_answer(
            answer_id=answer_id,
            query_id=req.query_id,
            answer=result["answer"],
            citations=citations_full,  # text 포함 (4단계 조회용)
            status="succeeded",
            confidence=None,
            disclaimer=disclaimer
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [generate_sync] 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"답변 생성 중 오류 발생: {str(e)}"
        )


# ===== 비동기 방식 =====

# 메모리 기반 답변 캐시 (프로덕션에서는 Redis 등 사용 권장)
_answer_cache: Dict[str, Dict[str, Any]] = {}


def generate_async(req: AnswerRequest) -> AnswerAsyncResponse:
    """
    비동기 방식 답변 생성 요청 (백그라운드 처리)
    
    Args:
        req: AnswerRequest (query_id, selected_categories)
    
    Returns:
        AnswerAsyncResponse (answer_id, status="processing", estimated_seconds)
    """
    try:
        answer_id = str(uuid.uuid4())
        
        # query_id로부터 질문 복원
        query = query_store.get_query(req.query_id)
        question = query.get("question", "") if query else ""
        
        if not question:
            raise HTTPException(
                status_code=404,
                detail=f"Query ID '{req.query_id}'를 찾을 수 없습니다."
            )
        
        # 초기 상태 저장
        _answer_cache[answer_id] = {
            "status": "processing",
            "query_id": req.query_id
        }
        
        # scope 추론
        scope = _infer_scope(req.selected_categories)
        
        # category_id를 label로 변환
        category_labels = []
        for cat_id in req.selected_categories:
            if cat_id in ["LAW_전체", "MANUAL_전체", "ALL_전체"]:
                category_labels.append("전체")
            else:
                label = cat_id.split("_", 1)[1] if "_" in cat_id else cat_id
                label = label.replace("_", " ")
                category_labels.append(label)
        
        # 백그라운드 스레드에서 처리
        def process():
            try:
                rag = get_rag_instance()
                result = rag.generate_answer(
                    question=question,
                    selected_categories=category_labels,
                    scope=scope
                )
                
                citations_full = result.get("citations", [])  # text 포함
                
                # API 응답용 citations (text 제외)
                from schemas.answers import Citation
                citations_api = [
                    Citation(
                        chunk_id=c["chunk_id"],
                        doc_title=c["doc_title"],
                        score=c["score"]
                    )
                    for c in citations_full
                ]
                
                # disclaimer 설정
                disclaimer = None
                if scope in ["law", "all"]:
                    disclaimer = "본 답변은 법률 자문이 아니며, 참고용으로만 사용하시기 바랍니다."
                
                # 완료 상태 저장 (캐시에는 API용 citations)
                _answer_cache[answer_id] = {
                    "status": "completed",
                    "query_id": req.query_id,
                    "answer": result["answer"],
                    "disclaimer": disclaimer,
                    "citations": [c.dict() for c in citations_api],  # API용
                    "created_at": datetime.now().isoformat()
                }
                
                # DB에도 저장 (이력 관리 - text 포함)
                answer_store.save_answer(
                    answer_id=answer_id,
                    query_id=req.query_id,
                    answer=result["answer"],
                    citations=citations_full,  # text 포함 (4단계 조회용)
                    status="completed",
                    confidence=None,
                    disclaimer=disclaimer
                )
            except Exception as e:
                # 실패 상태 저장
                _answer_cache[answer_id] = {
                    "status": "failed",
                    "query_id": req.query_id,
                    "error_message": str(e)
                }
        
        # 스레드 시작
        threading.Thread(target=process, daemon=True).start()
        
        return AnswerAsyncResponse(
            answer_id=answer_id,
            status="processing",
            estimated_seconds=10
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [generate_async] 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"답변 생성 요청 중 오류 발생: {str(e)}"
        )


def get_status(answer_id: str) -> AnswerStatusResponse:
    """
    비동기 답변 상태 조회 (polling용)
    
    Args:
        answer_id: 답변 ID
    
    Returns:
        AnswerStatusResponse (answer_id, status, answer, citations, created_at, error_message)
    """
    if answer_id not in _answer_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Answer ID '{answer_id}'를 찾을 수 없습니다."
        )
    
    data = _answer_cache[answer_id]
    
    return AnswerStatusResponse(
        answer_id=answer_id,
        status=data["status"],
        answer=data.get("answer"),
        disclaimer=data.get("disclaimer"),
        citations=data.get("citations"),
        created_at=data.get("created_at"),
        error_message=data.get("error_message")
    )


def get_answer_by_id(answer_id: str) -> AnswerResponse:
    """
    답변 ID로 답변 조회 (동기 API용)
    
    Args:
        answer_id: 답변 ID
    
    Returns:
        AnswerResponse
    """
    answer = answer_store.get_answer(answer_id)
    
    if not answer:
        raise HTTPException(
            status_code=404,
            detail=f"Answer ID '{answer_id}'를 찾을 수 없습니다."
        )
    
    return AnswerResponse(
        answer_id=answer["answer_id"],
        query_id=answer["query_id"],
        status=answer["status"],
        answer=answer["answer"],
        confidence=answer.get("confidence"),
        disclaimer=answer.get("disclaimer"),
        citations=answer["citations"],
        created_at=answer["created_at"]
    )


# ===== 유틸리티 =====

def _infer_scope(categories: list) -> str:
    """
    카테고리 ID 목록에서 scope 추론
    
    예: ["LAW_식품위생법", "MANUAL_HACCP관리"] → "all"
         ["LAW_식품위생법"] → "law"
         ["MANUAL_HACCP관리"] → "manual"
         ["LAW_전체"] → "law"
         ["ALL_전체"] → "all"
    """
    has_law = any(c.startswith("LAW_") for c in categories)
    has_manual = any(c.startswith("MANUAL_") for c in categories)
    has_all = any(c.startswith("ALL_") for c in categories)
    
    if has_all or (has_law and has_manual):
        return "all"
    elif has_law:
        return "law"
    elif has_manual:
        return "manual"
    else:
        return "all"

