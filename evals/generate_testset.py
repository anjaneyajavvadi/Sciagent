import app.env_setup
import json
import os
import pickle
import random
from app.tools.llm import LLM
from app.utils.logger import logger
from config import BM25_INDEX_PATH

def generate_testset(output_path:str="data/eval_data/testset.json",n:int=50):

    with open(BM25_INDEX_PATH,'rb') as f:
        data=pickle.load(f)
        chunks=data['chunks']
    
    logger.info(f"Loaded {len(chunks)} chunks")

    sampled=random.sample(chunks,min(n,len(chunks)))
    llm=LLM()
    testset=[]

    for i,chunk in enumerate(sampled):
        logger.info(f"Generating Q&A {i+1}/{len(sampled)}")
        try:
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research QA generator. "
                        "Given a passage from a research paper, generate one specific question "
                        "that can be answered using ONLY the passage, and provide the answer. "
                        "Return ONLY valid JSON in this exact format, nothing else:\n"
                        '{"question": "...", "answer": "..."}'
                    )
                },
                {
                    "role": "user",
                    "content": f"Passage:\n{chunk.text[:1000]}"
                }
            ]
            response=llm.chat(messages)
            clean=response.strip().replace("```json", "").replace("```", "").strip()
            qa=json.loads(clean)

            testset.append({
                "question":   qa["question"],
                "answer":     qa["answer"],
                "context":    chunk.text,
                "paper_id":   chunk.paper_id,
                "title":      chunk.title,
            })
        
        except Exception as e:
            logger.warning(f"Failed on chunk {i}: {e}")
            continue

    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    with open(output_path,"w") as f:
        json.dump(testset,f,indent=2)

    logger.info(f"Testset saved: {len(testset)} Q&A pairs → {output_path}")
    return testset

if __name__ == "__main__":
    generate_testset()