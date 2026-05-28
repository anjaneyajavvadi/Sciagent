import app.env_setup

from fastapi import FastAPI,Depends,HTTPException
from contextlib import asynccontextmanager
from app.api.schemas import QueryRequest,QueryResponse,HealthResponse
from app.api.dependencies import get_agent
from app.indexing.vector_store import VectorStore
from app.indexing.bm25_index import BM25Index
from app.utils.logger import logger
import uuid
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up - loading agent")
    get_agent()
    logger.info("Agent ready")
    yield
    logger.info("Shutting down")

app=FastAPI(
    title="SciAgent API",
    description="Agentic RAG over Arxiv research papers",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health",response_model=HealthResponse)
def health():
    try:
        vs=VectorStore()
        info=vs.collection_info()
        qdrant_status=f"OK- {info['vectors_count']} points"
    except Exception as e:
        qdrant_status = f"error: {str(e)}"

    try:
        bm25=BM25Index()
        bm25._load()
        bm25_status = f"OK — {len(bm25.chunks)} chunks"
    except Exception as e:
        bm25_status = f"error: {str(e)}"

    return HealthResponse(
        status='ok',
        qdrant=qdrant_status,
        bm25_index=bm25_status,
        collection = os.getenv("QDRANT_COLLECTION", "research_papers"),
    )


@app.post("/query",response_model=QueryResponse)
def query(request:QueryRequest,agent=Depends(get_agent)):
    logger.info(f"[api] query:{request.query}")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    thread_id=request.thread_id or str(uuid.uuid4())
    config={"configurable":{"thread_id":thread_id}}


    try:
        result = agent.invoke({
            "query":              request.query,
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
        },
        config=config)

        return QueryResponse(
            answer          = result["answer"],
            sources         = result["sources"],
            web_search_used = result["web_search_used"],
            sub_questions   = result["sub_questions"],
            iteration_count = result["iteration_count"],
            thread_id       = thread_id  
        )
    except Exception as e:
        logger.error(f"[api] query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))