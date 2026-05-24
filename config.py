"""
Central config. Every module imports from here — no magic strings scattered around.
"""
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "research_papers")

BGE_MODEL         = "BAAI/bge-m3"
DENSE_DIM         = 1024          
BATCH_SIZE        = 8             

CHUNK_SIZE        = 512           
CHUNK_OVERLAP     = 51            
TIKTOKEN_ENCODER="cl100k_base"
TOP_K_DENSE       = 20            
TOP_K_BM25        = 20
TOP_K_FINAL       = 5             

RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_THRESHOLD  = 0.0           

AZURE_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

DATA_DIR          = "data"
PAPERS_DIR        = f"{DATA_DIR}/papers"
INDEX_DIR         = f"{DATA_DIR}/indexes"
BM25_INDEX_PATH   = f"{INDEX_DIR}/bm25_index.pkl"

MAX_PAPERS        = 50    
ARXIV_TOPICS = {
    "artificial_intelligence":  "ti:artificial intelligence OR abs:large language model",
    "computer_vision":          "ti:computer vision OR abs:image recognition object detection",
    "nlp":                      "ti:natural language processing OR abs:text classification named entity",
    "statistics":               "ti:statistical learning OR abs:bayesian inference causal inference",
    "deep_learning":            "ti:deep learning OR abs:neural network transformer training",
}
 