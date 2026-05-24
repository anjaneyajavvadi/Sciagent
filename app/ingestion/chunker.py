from config import CHUNK_OVERLAP,CHUNK_SIZE,TIKTOKEN_ENCODER
from app.ingestion.arxiv_fetcher import PaperDoc
from app.utils.logger import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from dataclasses import dataclass
from typing import List

@dataclass
class Chunk:
    chunk_id:str
    paper_id:str
    title:str
    authors:List[str]
    published:str
    url:str
    chunk_index:int
    total_chunks:int
    text:str

    def to_dict(self)->dict:
        return {
            "chunk_id":     self.chunk_id,
            "paper_id":     self.paper_id,
            "title":        self.title,
            "authors":      self.authors,
            "published":    self.published,
            "url":          self.url,
            "chunk_index":  self.chunk_index,
            "total_chunks": self.total_chunks,
            "text":         self.text,
        }
    
class Chunker:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size        = CHUNK_SIZE,
            chunk_overlap     = CHUNK_OVERLAP,
            length_function   = len,
            separators        = ["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_docs(self,docs:List[PaperDoc])->List[Chunk]:
        logger.info(f"got {len(docs)} documents")
        all_chunks:List[Chunk]=[]

        for doc in docs:
            raw=self._clean_text(doc.full_text)
            header=f"Title: {doc.title}\n Abstract: {doc.abstract}\n\n"
            enriched=header+raw

            raw_chunks=self.splitter.split_text(enriched)

            if not raw_chunks:
                logger.warning(f"No chunks produced for {doc.paper_id}, skipping")
                continue

            doc_chunks:List=[]
            for i,text in enumerate(raw_chunks):
                chunk = Chunk(
                    chunk_id    = f"{doc.paper_id}__chunk_{i}",
                    paper_id    = doc.paper_id,
                    title       = doc.title,
                    authors     = doc.authors,
                    published   = doc.published,
                    url         = doc.url,
                    chunk_index = i,
                    total_chunks= 0,   # patched below
                    text        = text,
                )
                doc_chunks.append(chunk)

            for c in doc_chunks:
                c.total_chunks=len(doc_chunks)

            all_chunks.extend(doc_chunks)
            logger.info(f"{doc.paper_id}: {len(doc_chunks)} chunks")

        logger.info(f"Total chunks: {len(all_chunks)} from {len(docs)} papers")
        return all_chunks
    
    def _clean_text(self,text:str)->str:
        text=re.sub(r"\n{3,}","\n\n",text)
        lines=text.split("\n")
        cleaned=[]

        for line in lines:
            stripped=line.strip().lower()
            if len(stripped)<3:
                continue
            cleaned.append(stripped)
        return "\n".join(cleaned)
    
