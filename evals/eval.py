import app.env_setup
import json
import os
import pickle
import random
from app.tools.llm import LLM
from app.graph.graph import build_graph
from app.utils.logger import logger
from config import BM25_INDEX_PATH
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval import evaluate
from deepeval.models import DeepEvalBaseLLM


# ── Azure LLM wrapper for DeepEval ────────────────────────────────────────────
class AzureDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.llm = LLM()

    def load_model(self):
        return self.llm

    def generate(self, prompt: str) -> str:
        response = self.llm.chat_json([
            {
                "role":    "system",
                "content": "You are a JSON-only response bot. Always respond with valid JSON. No markdown, no explanation, no text outside JSON."
            },
            {
                "role":    "user",
                "content": prompt
            }
        ])
        clean = response.strip()
        clean = clean.replace("```json", "").replace("```", "").strip()
        clean = clean.replace("\\_", "_")
        clean = clean.replace("\\.", ".")
        clean = clean.replace("\\'", "'")
        return clean

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "gpt-4o"


# ── Run evaluation ────────────────────────────────────────────────────────────
def run_evaluation(
    testset_path: str   = "data/eval_data/testset.json",
    output_path:  str   = "data/eval_data/results.json",
    max_samples:  int   = 5,
    threshold:    float = 0.5,
):
    with open(testset_path) as f:
        data = json.load(f)
    testset = random.sample(data, max_samples)
    logger.info(f"Running DeepEval on {len(testset)} samples")

    agent      = build_graph()
    azure_llm  = AzureDeepEvalLLM()
    test_cases = []

    for i, sample in enumerate(testset):
        logger.info(f"Sample {i+1}/{len(testset)}: {sample['question'][:60]}")
        try:
            result = agent.invoke({
                "query":              sample["question"],
                "guardrail":          "",
                "sub_questions":      [],
                "retrieved_chunks":   [],
                "reranked_chunks":    [],
                "compressed_context": "",
                "reflection":         "",
                "answer":             "",
                "sources":            [],
                "web_search_used":    False,
                "iteration_count":    0,
            })

            test_cases.append(LLMTestCase(
                input             = sample["question"],
                actual_output     = result["answer"],
                retrieval_context = [r["payload"]["text"] for r in result["reranked_chunks"]],
                expected_output   = sample["answer"],
            ))

        except Exception as e:
            logger.warning(f"Sample {i} failed: {e}")
            continue

    logger.info(f"Evaluating {len(test_cases)} test cases...")

    metrics = [
        FaithfulnessMetric(threshold=threshold,       model=azure_llm, verbose_mode=True),
        AnswerRelevancyMetric(threshold=threshold,     model=azure_llm, verbose_mode=True),
        ContextualRecallMetric(threshold=threshold,    model=azure_llm, verbose_mode=True),
        ContextualPrecisionMetric(threshold=threshold, model=azure_llm, verbose_mode=True),
    ]

    results = evaluate(test_cases=test_cases, metrics=metrics)

    # ── extract scores immediately before DeepEval serialization ─────────────
    scores = {}
    for test_result in results.test_results:
        for metric_data in test_result.metrics_data:
            name = metric_data.name
            if name not in scores:
                scores[name] = []
            if metric_data.score is not None:
                scores[name].append(metric_data.score)

    final = {
        k: round(sum(v) / len(v), 4) if v else 0.0
        for k, v in scores.items()
    }
    final["num_samples"] = len(test_cases)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"DeepEval Results ({final['num_samples']} samples):")
    for k, v in final.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"{'='*50}")

    return final


if __name__ == "__main__":
    if not os.path.exists("data/eval_data/testset.json"):
        logger.info("Testset not found, generating...")

    run_evaluation(max_samples=20)