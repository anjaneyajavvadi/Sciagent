from config import MAX_PAPERS,PAPERS_DIR,ARXIV_TOPICS
import os
import json
import time
from dataclasses import dataclass,asdict
import arxiv
import fitz
from typing import List, Optional
from app.utils.logger import logger
from urllib.request import urlretrieve

@dataclass
class PaperDoc:
    paper_id:   str
    title:      str
    authors:    List[str]
    abstract:   str
    published:  str
    url:        str
    full_text:  str          
    categories: List[str]

class Fetcher:
    def __init__(self):
        self.max_papers=MAX_PAPERS
        self.data_dir=PAPERS_DIR

    def fetch_papers(self,query:str,max_results:int=MAX_PAPERS,save_dir:str=PAPERS_DIR)->List[PaperDoc]:
        client=arxiv.Client(
            page_size=min(max_results,100),
            delay_seconds=3,
            num_retries=3,
        )

        search=arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        docs:List[PaperDoc]=[]
        meta_path=os.path.join(save_dir,'metadata.jsonl')
        os.makedirs(save_dir,exist_ok=True)

        processed_ids:set=set()
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                for line in f:
                    try:
                        processed_ids.add(json.loads(line)['paper_id'])
                    except Exception:
                        pass
        
        new_meta: List[dict] = []

        for paper in client.results(search):
            paper_id=paper.get_short_id()

            if(paper_id in processed_ids):
                continue

            pdf_path=self._download_paper(paper,save_dir)
            full_text=""
            if pdf_path:
                full_text=self._extract_pdf_text(pdf_path)
            
            if len(full_text.strip()) < 200:
                full_text = paper.summary
            
            doc=PaperDoc(
                paper_id=paper_id,
                title=paper.title,
                authors=[a.name for a in paper.authors],
                abstract=paper.summary,
                published=paper.published.strftime("%Y-%m-%d"),
                url=paper.entry_id,
                full_text=full_text,
                categories=[c for c in paper.categories],
            )

            docs.append(doc)
            new_meta.append(asdict(doc))
            time.sleep(0.5)
        if new_meta:
            with open(meta_path, "a") as f:
                for m in new_meta:
                    m_slim = {k: v for k, v in m.items() if k != "full_text"}
                    f.write(json.dumps(m_slim) + "\n")
            logger.info(f"Saved metadata for {len(new_meta)} new papers to {meta_path}")
 
        logger.info(f"Fetched {len(docs)} papers total ({len(processed_ids)} already cached)")
        return docs

    def _extract_pdf_text(self,pdf_path:str)->str:
        try:
            doc=fitz.open(pdf_path)
            pages=[]
            for page in doc:
                pages.append(page.get_text("text"))
            doc.close()
            return "\n".join(pages)
        except Exception as e:
            return ""


    def _download_paper(self, paper: arxiv.Result, save_dir: str) -> Optional[str]:
        os.makedirs(save_dir, exist_ok=True)
        paper_id = paper.get_short_id().replace("/", "_")
        pdf_path = os.path.join(save_dir, f"{paper_id}.pdf")

        if os.path.exists(pdf_path):
            logger.info(f"[cache] {paper_id}.pdf already exists, skipping download")
            return pdf_path

        try:
            urlretrieve(paper.pdf_url, pdf_path)
            logger.info(f"[download] {paper_id}.pdf")
            return pdf_path

        except Exception as e:
            logger.warning(f"[fail] PDF download failed for {paper_id}: {e}")
        return None
        
    def fetch_all_topics(
        self,
        max_per_topic: int = 10,
        topics: dict = None,
        save_dir: str = PAPERS_DIR,
    ) -> List[PaperDoc]:
        """
        Fetch papers for every topic in ARXIV_TOPICS (or a custom dict).
    
        Args:
            max_per_topic: papers to fetch per topic. 10 topics × 10 papers = 100 max.
                        Keep low (5-10) during dev to avoid OOM on 8GB RAM.
            topics:        override ARXIV_TOPICS with a custom {label: query} dict.
            save_dir:      where to store PDFs and metadata.
    
        Returns:
            Deduplicated flat list of PaperDoc across all topics.
        """
        topics = topics or ARXIV_TOPICS
        all_docs: List[PaperDoc] = []
        seen_ids: set = set()
    
        for label, query in topics.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Topic: {label}")
            logger.info(f"Query: {query}")
            logger.info(f"{'='*50}")
    
            docs = self.fetch_papers(query, max_results=max_per_topic, save_dir=save_dir)
    
            added = 0
            for doc in docs:
                if doc.paper_id not in seen_ids:
                    seen_ids.add(doc.paper_id)
                    all_docs.append(doc)
                    added += 1
    
            logger.info(f"Topic '{label}': {added} new papers (total so far: {len(all_docs)})")
    
        logger.info(f"\nDone. Total unique papers: {len(all_docs)}")
        return all_docs
            
