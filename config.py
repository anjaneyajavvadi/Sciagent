"""
Central config. Every module imports from here — no magic strings scattered around.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "research_papers")

# ── Embedding ─────────────────────────────────────────────────────────────────
# BGE-M3 produces BOTH dense (1024-d) and sparse vectors in one forward pass.
# On your i5 it takes ~3-4 GB RAM. If it OOMs, swap to BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_MODEL         = "BAAI/bge-m3"
DENSE_DIM         = 1024           # BGE-M3 dense output dimension
BATCH_SIZE        = 4              # keep small to not OOM on 8GB

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE        = 512            # tokens
CHUNK_OVERLAP     = 51             # ~10%

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_DENSE       = 20             # fetch more, reranker will cut it down
TOP_K_BM25        = 20
TOP_K_FINAL       = 5              # what goes into the context window after rerank

# ── Reranker ──────────────────────────────────────────────────────────────────
# ~22MB download, runs on CPU in <500ms for 20 candidates
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_THRESHOLD  = 0.0            # drop candidates below this score

# ── Azure OpenAI ──────────────────────────────────────────────────────────────
AZURE_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR          = "data"
PAPERS_DIR        = f"{DATA_DIR}/papers"
INDEX_DIR         = f"{DATA_DIR}/indexes"
BM25_INDEX_PATH   = f"{INDEX_DIR}/bm25_index.pkl"

# ── Arxiv ─────────────────────────────────────────────────────────────────────
MAX_PAPERS        = 50    
ARXIV_TOPICS = {
    "artificial_intelligence":  "ti:artificial intelligence OR abs:large language model",
    "computer_vision":          "ti:computer vision OR abs:image recognition object detection",
    "nlp":                      "ti:natural language processing OR abs:text classification named entity",
    "statistics":               "ti:statistical learning OR abs:bayesian inference causal inference",
    "deep_learning":            "ti:deep learning OR abs:neural network transformer training",
}
 