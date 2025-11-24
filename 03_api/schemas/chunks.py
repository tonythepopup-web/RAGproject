"""
청크 상세 정보 스키마
"""
from pydantic import BaseModel


class ChunkDetailResponse(BaseModel):
    """청크 상세 정보 응답"""
    answer_id: str
    query_id: str
    chunk_id: str
    chunk_text: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer_id": "a_41f7e0",
                "query_id": "q_9b2d1c",
                "chunk_id": "c_ab34",
                "chunk_text": "식품위생법 제1조..."
            }
        }

