from app.retrieval.hybrid_retrieval import HybridRetriever
from app.utils.logger import logger
from sentence_transformers import CrossEncoder
from typing import List,Dict
from config import RERANKER_MODEL,TOP_K_FINAL,RERANK_THRESHOLD



class Reranker:
    def __init__(self):
        logger.info(f"Loading Reranker: {RERANKER_MODEL}")
        self.model = CrossEncoder(RERANKER_MODEL)
        logger.info("Reranker loaded successfully")

    def rerank(self,query:str,results:List[Dict],top_k:int=TOP_K_FINAL)->List[Dict]:
        if not results:
            logger.warning("Reranker recieved empty results")
            return []

        pairs=[(query,r['payload']['text']) for r in results]
        scores=self.model.predict(pairs)

        scored=[
            {"score":float(score),"payload":result['payload']}
            for score,result in zip(scores,results)
        ]

        scored = [r for r in scored if r["score"] > RERANK_THRESHOLD]
        scored = sorted(scored, key=lambda x: x["score"], reverse=True)

        logger.info(f"Reranker: {len(results)} in → {len(scored[:top_k])} out")

        return scored[:top_k]

