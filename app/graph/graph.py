import app.env_setup
from app.graph.state import AgentState
from app.graph.nodes import build_nodes, should_web_search, should_retry,is_relevant
from app.indexing.vector_store import VectorStore
from app.ingestion.embedder import Embedder
from app.indexing.bm25_index import BM25Index
from app.retrieval.hybrid_retrieval import HybridRetriever
from app.retrieval.reranker import Reranker
from app.utils.logger import logger
from langgraph.graph import StateGraph, END


def build_graph():
    vector_store = VectorStore()
    embedder     = Embedder()
    bm25         = BM25Index()
    retriever    = HybridRetriever(vector_store, embedder, bm25)
    reranker     = Reranker()
    nodes        = build_nodes(retriever, reranker)

    graph = StateGraph(AgentState)

    graph.add_node("guardrail",nodes['guardrail'])
    graph.add_node("planner",nodes["planner"])
    graph.add_node("retrieve",nodes["retrieve"])
    graph.add_node("web_search",nodes["web_search"])
    graph.add_node("rerank",nodes["rerank"])
    graph.add_node("compress",nodes["compress"])
    graph.add_node("reflect",nodes["reflect"])
    graph.add_node("generate",nodes["generate"])
    graph.add_node("reject",nodes['reject'])
    graph.add_node("replan",nodes['replan'])
    

    graph.set_entry_point("guardrail")

    graph.add_conditional_edges(
        "guardrail",
        is_relevant,
        {"planner":"planner","reject":"reject"}
    )

    graph.add_edge("planner", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges(
        "rerank",
        should_web_search,
        {"web_search": "web_search", "compress":"compress"}
    )

    graph.add_edge("web_search", "compress")
    graph.add_edge("compress",   "reflect")

    graph.add_conditional_edges(
        "reflect",
        should_retry,
        {"replan": "replan", "generate": "generate"}
    )
    graph.add_edge("replan", "retrieve")

    graph.add_edge("generate", END)

    return graph.compile()


if __name__ == "__main__":
    agent = build_graph()

    result = agent.invoke({
        "query":              "Tell me about how currently reseacrh is going on moon and in space by NASA and ISRO",
        "guardrail":          "",
        "sub_questions":      [],
        "retrieved_chunks":   [],
        "reranked_chunks":    [],
        "web_chunks":         [],
        "compressed_context": "",
        "reflection":         "",
        "answer":             "",
        "sources":            [],
        "web_search_used":    False,
        "iteration_count":    0,
    })

    print(f"\n{'='*60}")
    print(f"Answer:\n{result['answer']}")
    print(f"\nSources:")
    for s in result["sources"]:
        print(f"  - {s}")
    print(f"Web search used: {result['web_search_used']}")