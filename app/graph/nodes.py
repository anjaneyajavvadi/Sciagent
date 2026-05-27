from config import AZURE_API_KEY,AZURE_DEPLOYMENT,AZURE_ENDPOINT,TOP_K_FINAL
from app.graph.state import AgentState
from app.retrieval.hybrid_retrieval import HybridRetriever
from app.retrieval.reranker import Reranker
from app.utils.logger import logger
from typing import List,Dict,Any
from app.tools.llm import LLM
from app.tools.tavily_client import TavilySearch
import os

SYSTEM_PROMPT = """You are a research assistant specialized in academic papers.
Answer the user's question using ONLY the provided context.
Always cite the paper title and arxiv ID for every claim you make.
If the context does not contain enough information, say so clearly.
Do not hallucinate or use outside knowledge."""

def build_nodes(retriever:HybridRetriever,reranker:Reranker):
    web_client=TavilySearch()
    llm=LLM()

    def guardrail_node(state:AgentState)->AgentState:
        logger.info(f"[guardrail] checking query: {state['query']}")
        messages=[
            {
                'role':"system",
                'content':(
                "You are a query validator for a research paper assistant. "
                "Determine if the user query is related to academic research, "
                "science, technology, machine learning, statistics, or any scholarly topic. "
                "Reply with exactly one of:\n"
                "RELEVANT: <one line reason>\n"
                "IRRELEVANT: <one line reason>"
                )
            },
            {
            "role": "user",
            "content": state["query"]
            }
        ]
        response=llm.chat(messages)
        logger.info(f"[guardrail] {response.strip()}")
        return {**state, "guardrail": response.strip()}

    def reject_node(state:AgentState)->AgentState:
        logger.info("[reject] query not relevant to research")
        return {
            **state,
            "answer": "I am a research assistant and can only answer questions related to academic papers, science, technology, or scholarly topics. Please ask a research-related question.",
            "sources":[]
        }

    def planner_node(state:AgentState)->AgentState:
        logger.info(f"[planner] decomposing query: {state['query']}")
        messages = [
            {
                "role":    "system",
                "content": (
                    "You are a research query planner. "
                    "Break the user query into 2-3 specific sub-questions "
                    "that together fully answer the original question. "
                    "Return ONLY a numbered list, nothing else. Example:\n"
                    "1. What is X?\n2. How does Y work?\n3. What are the limitations of Z?"
                )
            },
            {
                "role":    "user",
                "content": state["query"]
            }
        ]
        response=llm.chat(messages)
        lines=response.strip().split('\n')
        sub_questions = [l.split(". ", 1)[-1].strip() for l in lines if l.strip()]
        logger.info(f"[planner] sub-questions: {sub_questions}")
        return {**state, "sub_questions": sub_questions}

    def retrieve_node(state:AgentState)->AgentState:
        logger.info(f"[retrieve] query: {state['query']}")
        all_chunks=[]
        seen_ids=set()

        queries=[state["query"]]+state.get("sub_questions",[])
        for q in queries:
            results=retriever.retrieve(q,top_k=10)
            for r in results:
                cid=r['payload']['chunk_id']
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.append(r)
        logger.info(f"[retrieve] {len(all_chunks)} unique chunks across {len(queries)} queries")
        return {
            **state,
            "retrieved_chunks": all_chunks,
            "web_search_used":  False,
        }

    def web_search_node(state: AgentState) -> AgentState:
        logger.info(f"[web_search] query: {state['query']}")
        web_chunks = web_client.search(state["query"])
        merged     = state["retrieved_chunks"] + web_chunks
        logger.info(f"[web_search] added {len(web_chunks)} web chunks")
        return {**state, "retrieved_chunks": merged, "web_search_used": True}
    

    def rerank_node(state: AgentState) -> AgentState:
        logger.info(f"[rerank] reranking {len(state['retrieved_chunks'])} chunks")
        reranked = reranker.rerank(
            query   = state["query"],
            results = state["retrieved_chunks"],
            top_k   = TOP_K_FINAL,
        )
        logger.info(f"[rerank] {len(reranked)} chunks after rerank")
        return {**state, "reranked_chunks": reranked}
    
    def compress_node(state: AgentState) -> AgentState:
        logger.info("[compress] building context window")
        chunks  = state["reranked_chunks"]
        sources = []
        parts   = []

        for i, r in enumerate(chunks):
            p        = r["payload"]
            title    = p.get("title", "Unknown")
            paper_id = p.get("paper_id", "")
            text     = p.get("text", "")
            url      = p.get("url", "")

            parts.append(f"[{i+1}] {title} ({paper_id})\n{text}")
            source = f"{title} — {url}" if url else title
            if source not in sources:
                sources.append(source)

        compressed = "\n\n".join(parts)
        logger.info(f"[compress] {len(compressed)} chars, {len(sources)} sources")
        return {**state, "compressed_context": compressed, "sources": sources}
    

    def reflect_node(state: AgentState) -> AgentState:
        logger.info("[reflect] evaluating context quality")
        response = llm.chat([
            {
                "role": "system",
                "content": (
                    "You are a research quality checker. "
                    "Given a question and retrieved context, decide if the context "
                    "has SOME relevant information to partially or fully answer the question. "
                    "Be generous — if the context contains anything relevant, mark it SUFFICIENT. "
                    "Only mark INSUFFICIENT if the context is completely unrelated. "
                    "Reply with exactly one of:\n"
                    "SUFFICIENT: <one line reason>\n"
                    "INSUFFICIENT: <one line reason>"
                )
            },
            {
                "role": "user",
                "content": f"Question: {state['query']}\n\nContext:\n{state['compressed_context'][:2000]}"
            }
        ])
        logger.info(f"[reflect] {response.strip()}")
        return {**state, "reflection": response.strip()}

    def replan_node(state:AgentState)->AgentState:
        messages=[
            {
                "role":"system",
                "content":"The previous retrieval was insufficient. Rephrase the query differently to find better results. Return only the rephrased query, nothing else."
            },
            {
                "role":"user",
                "content":(
                    f"Original query:{state['query']}\n",
                    f"Why it failed:{state['reflection']}\n"
                    f"Generate a better search query."
                )
            }
        ]
        response=llm.chat(messages)
        return {**state, "query": response.strip(), "iteration_count": state["iteration_count"] + 1}
    
    
    def generate_node(state: AgentState) -> AgentState:
        logger.info("[generate] calling Azure OpenAI")
        messages = [
        {
            "role":    "system",
            "content": (
                SYSTEM_PROMPT
            )
        },
        {
            "role":    "user",
            "content": (
                f"Question: {state['query']}\n\n"
                f"Context:\n{state['compressed_context']}"
            )
        }
        ]
        response = llm.chat(messages)
        logger.info(f"[generate] answer: {len(response)} chars")
        return {**state, "answer": response}
    return {
        "planner":    planner_node,
        "retrieve":   retrieve_node,
        "web_search": web_search_node,
        "rerank":     rerank_node,
        "compress":   compress_node,
        "reflect":    reflect_node,
        "generate":   generate_node,
        "guardrail":  guardrail_node,
        "reject":     reject_node,
        "replan":   replan_node,
    }



def should_web_search(state: AgentState) -> str:
    chunks = state["retrieved_chunks"]

    if not chunks:
        return "web_search"

    top_scores = sorted(
        [r["score"] for r in chunks],
        reverse=True
    )[:5]
    avg_score = sum(top_scores) / len(top_scores)

    logger.info(f"[router] avg top-5 RRF score: {avg_score:.4f}")

    if avg_score < 0.02:
        logger.info("[router] low relevance scores → web search")
        return "web_search"

    return "rerank"

def should_retry(state:AgentState)->str:
    reflection=state.get('reflection',"")
    iteration_count=state.get("iteration_count",0)

    if "INSUFFICIENT" in reflection and iteration_count<2:
        logger.info(f"[router] context insufficient, retrying (attempt {iteration_count + 1})")
        return "replan"
    logger.info("[router] context sufficient → generate")
    return "generate"

def is_relevant(state: AgentState) -> str:
    if "IRRELEVANT" in state.get("guardrail", ""):
        logger.info("[router] irrelevant query → reject")
        return "reject"
    return "planner"


def should_retry(state: AgentState) -> str:
    if "INSUFFICIENT" in state.get("reflection", "") and state.get("iteration_count", 0) < 2:
        return "replan"   
    return "generate"