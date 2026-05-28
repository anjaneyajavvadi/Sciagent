from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer  :        str
    sources :        List[str]
    web_search_used: bool
    sub_questions:   List[str]
    iteration_count: int

class HealthResponse(BaseModel):
    status:     str
    qdrant:     str
    bm25_index: str
    collection: str