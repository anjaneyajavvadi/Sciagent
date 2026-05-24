from config import TOP_K_BM25,TOP_K_DENSE,TOP_K_FUSION
from app.ingestion.bm25_index import BM25Index
from app.ingestion.embedder import Embedder
from app.ingestion.qdrant_client import VectorStore
from app.utils.logger import logger
from typing import List,Dict

class HybridRetriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder, bm25: BM25Index):
        self.vector_store=vector_store
        self.embedder=embedder
        self.bm25=bm25

    def retrieve(self,query:str, top_k:int=TOP_K_FUSION)->List[Dict]:
        query_embedding=self.embedder.embed_query(query)

        dense_results=self.vector_store.dense_search(
            query_dense=query_embedding['dense'],
            top_k=TOP_K_DENSE
        )
        sparse_results=self.vector_store.sparse_search(
            query_sparse=query_embedding['sparse'],
            top_k=TOP_K_DENSE
        )
        bm25_results=self.bm25.search(query,top_k=TOP_K_BM25)

        fused=self._rrf_fusion([dense_results,sparse_results,bm25_results])

        logger.info(f"Hybrid retrieval: {len(dense_results)} dense, "
                    f"{len(sparse_results)} sparse, "
                    f"{len(bm25_results)} bm25 → {len(fused[:top_k])} after RRF")

        return fused[:top_k]
    
    def _rrf_fusion(self,
                    result_lists:List[List[Dict]],
                    k:int=60)->List[Dict]:
        scores:Dict[str,float]={}
        payloads:Dict[str,Dict]={}

        for result_list in result_lists:
            for rank,result in enumerate(result_list):
                payload  = dict(result["payload"])  
                if not payload or "chunk_id" not in payload:
                    logger.warning(f"Skipping result with empty/invalid payload: {result}")
                    continue
                chunk_id = payload["chunk_id"]
                scores[chunk_id]   = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
                payloads[chunk_id] = payload
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

        return [
            {"score": scores[cid], "payload": payloads[cid]}
            for cid in sorted_ids
        ]