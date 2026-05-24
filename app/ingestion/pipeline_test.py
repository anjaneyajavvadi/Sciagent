from app.ingestion.arxiv_fetcher import Fetcher,PaperDoc
from app.ingestion.chunker import Chunker
from app.ingestion.embedder import Embedder
from app.ingestion.qdrant_client import VectorStore
from app.ingestion.bm25_index import BM25Index
import json
import os
import fitz
from config import PAPERS_DIR

# def load_docs_from_disk() -> list:
#     meta_path = os.path.join(PAPERS_DIR, "metadata.jsonl")
#     docs = []

#     with open(meta_path) as f:
#         for line in f:
#             m = json.loads(line)
#             paper_id_clean = m["paper_id"].replace("/", "_")
#             pdf_path = os.path.join(PAPERS_DIR, f"{paper_id_clean}.pdf")

#             # read full text from already downloaded PDF
#             full_text = ""
#             if os.path.exists(pdf_path):
#                 try:
#                     doc = fitz.open(pdf_path)
#                     pages = [page.get_text("text") for page in doc]
#                     doc.close()
#                     full_text = "\n".join(pages)
#                 except Exception as e:
#                     print(f"Failed to read {pdf_path}: {e}")
            
#             # fallback to abstract if PDF unreadable
#             if len(full_text.strip()) < 200:
#                 print(f"[fallback] {m['paper_id']} using abstract")
#                 full_text = m["abstract"]

#             docs.append(PaperDoc(
#                 paper_id   = m["paper_id"],
#                 title      = m["title"],
#                 authors    = m["authors"],
#                 abstract   = m["abstract"],
#                 published  = m["published"],
#                 url        = m["url"],
#                 full_text  = full_text,
#                 categories = m["categories"],
#             ))

#     print(f"Loaded {len(docs)} docs from disk")
#     return docs


fetcher= Fetcher()
chunker= Chunker()
embedder= Embedder()
vector_store = VectorStore()
bm25 = BM25Index()

# docs= load_docs_from_disk()
docs=Fetcher.fetch_all_topics(max_per_topic=10)
chunks= chunker.chunk_docs(docs)
bm25.build(chunks)
embeddings = embedder.embed_chunks(chunks)
total= vector_store.upsert_chunks(chunks, embeddings)

print(f"Done. {total} points in Qdrant")
print(vector_store.collection_info())