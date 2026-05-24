from app.ingestion.qdrant_client import VectorStore
from app.ingestion.embedder import Embedder
from app.ingestion.bm25_index import BM25Index
from app.retrieval.hybrid_retrieval import HybridRetriever
from app.retrieval.reranker import Reranker

vectorstore=VectorStore()
embedder=Embedder()
bm25=BM25Index()
retriever=HybridRetriever(vectorstore,embedder,bm25)
reranker=Reranker()

query="what is attention mechanism in transformers"
results=retriever.retrieve(query,top_k=10)
ranked=reranker.rerank(query,results,top_k=5)

for i,r in enumerate(ranked):
    print(f"\n--- Result {i+1} | score: {r['score']:.4f}")
    print(f"Chunk_id:{r['payload']['chunk_id']}")
    print(f"Title: {r['payload']['title']}")
    print(f"Text:  {r['payload']['text'][:200]}")
