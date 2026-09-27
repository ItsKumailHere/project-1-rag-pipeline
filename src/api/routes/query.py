from fastapi import APIRouter, Depends
from src.models.schemas import QueryRequest, RAGResponse
from src.services.retrieval import RetrievalService
from src.services.generation import GenerationService

router = APIRouter()

def get_retrieval_service():
    return RetrievalService()

def get_generation_service():
    return GenerationService()

@router.post("/query", response_model=RAGResponse)
def query_rag(
    request: QueryRequest,
    retrieval_svc: RetrievalService = Depends(get_retrieval_service),
    generation_svc: GenerationService = Depends(get_generation_service)
):
    """
    Main RAG endpoint. 
    1. Retrieves relevant context using Hybrid Search (Vector + Sparse).
    2. Generates an answer bound to a strict Pydantic schema with citations.
    """
    # 1. Retrieve
    context = retrieval_svc.search(
        query=request.query, 
        top_k=request.top_k, 
        hybrid=request.hybrid
    )
    
    # 2. Generate
    response = generation_svc.generate_answer(
        query=request.query,
        context=context
    )
    
    return response
