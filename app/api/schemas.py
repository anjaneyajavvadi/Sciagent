from pydantic import BaseModel
from typing import List,Optional

class QueryRequest(BaseModel):
    query:     str
    thread_id: Optional[str]=None

class QueryResponse(BaseModel):
    answer  :        str
    sources :        List[str]
    web_search_used: bool
    sub_questions:   List[str]
    iteration_count: int
    thread_id:       str

class HealthResponse(BaseModel):
    status:     str
    qdrant:     str
    bm25_index: str
    collection: str