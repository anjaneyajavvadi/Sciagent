from typing import List, Dict, Any
from typing_extensions import TypedDict


class AgentState(TypedDict):
    query:              str
    sub_questions:      List[str]        
    retrieved_chunks:   List[Dict[str, Any]]
    reranked_chunks:    List[Dict[str, Any]]
    compressed_context: str
    reflection:         str              
    answer:             str
    sources:            List[str]
    web_search_used:    bool
    iteration_count:    int              