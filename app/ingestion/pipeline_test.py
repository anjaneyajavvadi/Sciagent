from app.ingestion.arxiv_fetcher import Fetcher
from app.ingestion.chunker import Chunker
from app.ingestion.embedder import Embedder
from app.ingestion.qdrant_client import VectorStore

fetcher= Fetcher()
chunker= Chunker()
embedder= Embedder()
vector_store = VectorStore()

docs= fetcher.fetch_all_topics(max_per_topic=10)
chunks= chunker.chunk_docs(docs)
embeddings = embedder.embed_chunks(chunks)
total= vector_store.upsert_chunks(chunks, embeddings)

print(f"Done. {total} points in Qdrant")
print(vector_store.collection_info())