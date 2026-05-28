from tavily import TavilyClient
from config import TAVILY_API_KEY
from typing import List,Dict,Any
from app.utils.logger import logger
import json

class TavilySearch:
    def __init__(self):
        self.client=TavilyClient(TAVILY_API_KEY)
    
    def search(self,query:str)->List[Dict[str,Any]]:
        results=self.client.search(
            query=query,
            max_results=5
        )

        web_chunks   = [
            {
                "score":   r.get("score"),
                "payload": {
                    "chunk_id": f"web_{i}",
                    "paper_id": "web",
                    "title":    r.get("title", "Web Result"),
                    "url":      r.get("url", ""),
                    "text":     r.get("content", ""),
                    "authors":  [],
                    "published": "",
                }
            }
            for i, r in enumerate(results.get("results", []))
        ]
        
        return web_chunks
    
if __name__=='__main__':
    client=TavilySearch()
    results=client.search("Latest developments in NASA and ISRO's lunar exploration missions and space research initiatives"
                  )
    logger.info(json.dumps(results, indent=2))
