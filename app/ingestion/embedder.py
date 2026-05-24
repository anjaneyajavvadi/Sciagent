from config import BGE_MODEL,BATCH_SIZE,MODEL_CACHE_DIR
from app.ingestion.chunker import Chunk
from app.utils.logger import logger
from FlagEmbedding import BGEM3FlagModel
from typing import List,Dict,Any
import numpy as np

class Embedder:
    def __init__(self):
        logger.info(f"Loading BGE-M3 model:{BGE_MODEL}")
        logger.info("First run downloads ~570MB, takes 2-5 min")
        self.model=BGEM3FlagModel(BGE_MODEL,use_fp16=False,cache_dir=MODEL_CACHE_DIR)
        logger.info("BGE-M3 loaded successfully")

    def embed_chunks(self,chunks:List[Chunk])->List[Dict[str,Any]]:
        texts=[chunk.text for chunk in chunks]
        results=[]

        for i in range(0,len(texts),BATCH_SIZE):
            batch=texts[i:i+BATCH_SIZE]
            logger.info(f"Embedding batch {i // BATCH_SIZE + 1} / {len(texts) // BATCH_SIZE + 1}")
            batch_results = self._embed_texts(batch)
            results.extend(batch_results)

        logger.info(f"Embedded {len(results)} chunks")
        return results

    def embed_query(self,query:str)->Dict[str,Any]:
        return self._embed_texts([query])[0]
    
    def _embed_texts(self,texts:List[str])->List[Dict[str,Any]]:
        output=self.model.encode(
            texts,batch_size=BATCH_SIZE,max_length=512,return_dense=True,return_sparse=True,return_colbert_vecs=False
        )
        dense_vecs=output['dense_vecs']
        lexical_weights=output['lexical_weights']

        results=[]
        for i in range(len(texts)):
            results.append({
                "dense":dense_vecs[i].astype(np.float32),
                "sparse":dict(lexical_weights[i])
            })
        
        return results