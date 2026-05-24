from langgraph.graph import MessagesState
from typing import List,Dict,Any
from typing_extensions import TypedDict


class AgentState:
    query:str
    retrieved_chunks:List[Dict[str,Any]]
    reranked_chunks:List[Dict[str,Any]]
    compresses_context:str
    answer:str
    sources:List[str]
    web_search_used:bool