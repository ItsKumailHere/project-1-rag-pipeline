from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Dict
from src.services.evaluation import EvaluationService
from src.services.retrieval import RetrievalService
from src.services.generation import GenerationService

router = APIRouter()

class EvalTestCase(BaseModel):
    question: str
    ground_truth: str = None

class EvalRequest(BaseModel):
    test_cases: List[EvalTestCase]

def get_services():
    return EvaluationService(), RetrievalService(), GenerationService()

@router.post("/evaluate")
def run_evaluation(
    request: EvalRequest,
    services = Depends(get_services)
):
    eval_svc, retrieval_svc, gen_svc = services
    
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    
    for case in request.test_cases:
        # Run the full RAG pipeline for the test case
        context = retrieval_svc.search(case.question, top_k=3, hybrid=False)
        response = gen_svc.generate_answer(case.question, context)
        
        questions.append(case.question)
        answers.append(response.answer)
        contexts_list.append([c.content for c in context])
        
        if case.ground_truth:
            ground_truths.append([case.ground_truth])
            
    # Run quantitative evaluation
    metrics = eval_svc.evaluate_batch(
        questions=questions,
        answers=answers,
        contexts=contexts_list,
        ground_truths=ground_truths if ground_truths else None
    )
    
    return {"metrics": metrics}
