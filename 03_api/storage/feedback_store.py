"""
피드백 저장소
"""
import json
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

# (answer_id, chunk_id) → list of feedbacks
_feedbacks: Dict[Tuple[str, str], List[Dict]] = {}

# Triplet 로그 저장 경로 (프로젝트 루트 기준 절대 경로)
# feedback_store.py -> storage/ -> 03_api/ -> 프로젝트루트/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRIPLET_LOG_PATH = PROJECT_ROOT / "00_data" / "output" / "training_data" / "triplets_group_bgem3.jsonl"


def save_feedback(
    answer_id: str,
    query_id: str,
    chunk_id: str,
    score: float,
    user_id: str = None,
    session_id: str = None
) -> None:
    """피드백 저장"""
    key = (answer_id, chunk_id)
    
    if key not in _feedbacks:
        _feedbacks[key] = []
    
    _feedbacks[key].append({
        "answer_id": answer_id,
        "query_id": query_id,
        "chunk_id": chunk_id,
        "score": score,
        "user_id": user_id,
        "session_id": session_id,
        "created_at": datetime.now().isoformat()
    })


def get_feedbacks(answer_id: str, chunk_id: str) -> List[Dict]:
    """특정 청크의 모든 피드백 조회"""
    key = (answer_id, chunk_id)
    return _feedbacks.get(key, [])


def compute_metrics(answer_id: str) -> Dict:
    """답변 전체의 피드백 메트릭 계산"""
    answer_feedbacks = [
        fb for key, fbs in _feedbacks.items()
        if key[0] == answer_id
        for fb in fbs
    ]
    
    if not answer_feedbacks:
        return {"avg_chunk_score": 0.0, "total_feedbacks": 0}
    
    total_score = sum(fb["score"] for fb in answer_feedbacks)
    avg_score = total_score / len(answer_feedbacks)
    
    return {
        "avg_chunk_score": round(avg_score, 2),
        "total_feedbacks": len(answer_feedbacks)
    }


def save_triplet_log(
    query: str,
    positives: List[str],
    negatives: List[str],
    pos_sources: Optional[List[str]] = None,
    neg_sources: Optional[List[str]] = None,
    extra_meta: Optional[Dict] = None
) -> None:
    """
    Triplet 로그를 JSONL 파일에 저장
    
    Args:
        query: 질문
        positives: 긍정 문서 리스트
        negatives: 부정 문서 리스트
        pos_sources: 긍정 문서 출처
        neg_sources: 부정 문서 출처
        extra_meta: 추가 메타데이터
    """
    def _clean(s: str) -> str:
        """텍스트 정리 (개행, 탭 제거)"""
        s = s.replace("\t", " ").replace("\r", " ").replace("\n", " ")
        return re.sub(r"\s+", " ", s).strip()
    
    # 레코드 생성
    rec = {
        "query": _clean(query),
        "positives": [_clean(p) for p in positives if p and p.strip()],
        "negatives": [_clean(n) for n in negatives if n and n.strip()],
        "meta": {"timestamp": datetime.now().isoformat(timespec="seconds")}
    }
    
    if pos_sources:
        rec["meta"]["pos_sources"] = [_clean(x) for x in pos_sources]
    if neg_sources:
        rec["meta"]["neg_sources"] = [_clean(x) for x in neg_sources]
    if extra_meta:
        rec["meta"].update(extra_meta)
    
    # 파일 저장 (디렉토리 자동 생성)
    TRIPLET_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(TRIPLET_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"✅ [Triplet 로그] 저장 완료: {TRIPLET_LOG_PATH}")

