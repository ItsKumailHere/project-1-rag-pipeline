from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
from src.services.ingestion import IngestionService

router = APIRouter()

class IngestRequest(BaseModel):
    document_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None

class IngestResponse(BaseModel):
    status: str
    message: str

def get_ingestion_service():
    return IngestionService()

@router.post("/ingest", response_model=IngestResponse)
def ingest_document(
    request: IngestRequest,
    ingestion_svc: IngestionService = Depends(get_ingestion_service)
):
    """
    Ingests a raw document into the RAG system.
    Chunks the text, embeds it, and stores it in pgvector.
    """
    ingestion_svc.process_and_store(
        text=request.text,
        document_id=request.document_id,
        metadata=request.metadata
    )
    
    return IngestResponse(status="success", message=f"Document {request.document_id} ingested successfully.")
