from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..db.models import ProcessingStatus

class DocumentBase(BaseModel):
    filename: str

class DocumentCreate(DocumentBase):
    pass

class DocumentResponse(DocumentBase):
    id: int
    upload_date: datetime
    status: ProcessingStatus
    page_count: Optional[int] = None
    file_size: Optional[int] = None

    class ConfigDict:
        from_attributes = True

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)

class SourceMetadata(BaseModel):
    filename: str
    page_number: int
    chunk_id: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceMetadata]
    context_used: bool
