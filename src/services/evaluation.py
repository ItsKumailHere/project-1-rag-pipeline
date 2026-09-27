import logging
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision
)

logger = logging.getLogger(__name__)

class EvaluationService:
    """
    Service to run quantitative RAG evaluation.
    This demonstrates the capability to rigorously evaluate LLM outputs,
    a critical skill for production AI engineering.
    """
    
    def __init__(self):
        # Ragas uses Langchain under the hood, so it respects OPENAI_API_KEY
        # If using OpenRouter, you may need custom LLM wrappers for Ragas,
        # but the architectural pattern holds.
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision
        ]
        
    def evaluate_batch(self, questions: List[str], answers: List[str], contexts: List[List[str]], ground_truths: List[List[str]] = None) -> Dict[str, float]:
        """
        Evaluates a batch of queries against the RAG system.
        """
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        
        if ground_truths:
            data["ground_truth"] = ground_truths
            
        dataset = Dataset.from_dict(data)
        
        logger.info(f"Running evaluation on {len(questions)} samples...")
        result = evaluate(
            dataset,
            metrics=self.metrics
        )
        
        return result
