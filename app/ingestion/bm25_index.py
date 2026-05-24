from config import BM25_INDEX_PATH,TOP_K_BM25
from app.ingestion.chunker import Chunk
from app.utils.logger import logger
from rank_bm25 import BM25Okapi
from typing import List,Dict
import pickle
import os
import re

class BM25Index:
    def __init__(self):
        self.bm25=None
        self.chunks=[]
        self.index_path=BM25_INDEX_PATH
    
    def build(self,chunks:List[Chunk]):
        logger.info(f"Building BM25 index on {len(chunks)} chunks")
        self.chunks=chunks
        tokenized=[self._tokenize(c.text) for c in chunks]
        self.bm25=BM25Okapi(tokenized)
        self._save()
        logger.info(f"BM25 index built and saved to {self.index_path}")

    def search(self,query:str,top_k:int=TOP_K_BM25)->List[Dict]:
        if self.bm25 is None:
            self._load()

        tokens=self._tokenize(query)
        scores=self.bm25.get_scores(tokens)

        scored=sorted(
            zip(scores,self.chunks),
            key=lambda x:x[0],
            reverse=True
        )[:top_k]

        return [
            {
                "score":float(score),"payload":chunk.to_dict()
            }
            for score,chunk in scored if score>0.0
        ]
    
    def _tokenize(self,query:str)->List[str]:
        text=query.lower()
        text=re.sub(r"[^a-z0-9\s]"," ",text)
        return text.split()
    
    def _save(self):
        os.makedirs(os.path.dirname(self.index_path),exist_ok=True)
        with open(self.index_path,'wb') as f:
            pickle.dump({"bm25":self.bm25, "chunks":self.chunks},f)
        logger.info(f"BM25 index saved: {self.index_path}")

    def _load(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"BM25 index not found at {self.index_path}. Run ingest_run.py first."
            )

        with open(self.index_path,'rb') as f:
            data=pickle.load(f)
            self.bm25=data['bm25']
            self.chunks=data['chunks']
        logger.info(f"BM25 index loaded: {len(self.chunks)} chunks")