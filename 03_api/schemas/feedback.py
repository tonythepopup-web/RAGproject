"""
청크 피드백 스키마
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class ChunkFeedback(BaseModel):
    """개별 청크 피드백"""
    chunk_id: str
    score: float = Field(..., ge=0.0, le=5.0, description="평점 (0.0 ~ 5.0)")


class ChunkFeedbackMeta(BaseModel):
    """피드백 메타 정보"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ChunkFeedbackRequest(BaseModel):
    """청크 피드백 요청 (여러 청크 한 번에 전송)"""
    answer_id: str
    query_id: str
    feedback: List[ChunkFeedback] = Field(..., description="청크별 평가 목록")
    meta: Optional[ChunkFeedbackMeta] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer_id": "a_41f7e0",
                "query_id": "q_9b2d1c",
                "feedback": [
                    {"chunk_id": "c_ab34", "feedback": "positive"},
                    {"chunk_id": "c_ac54", "feedback": "negative"}
                ],
                "meta": {
                    "user_id": "heewoo",
                    "session_id": "ui-20251021-001"
                }
            }
        }


class ChunkFeedbackResponse(BaseModel):
    """청크 피드백 응답 (사용 안 함 - 204 No Content 반환)"""
    pass

