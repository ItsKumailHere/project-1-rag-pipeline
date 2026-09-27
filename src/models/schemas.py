from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class DocumentChunk(BaseModel):
    id: str
    document_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class SearchResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]

class Citation(BaseModel):
    document_id: str = Field(description="ID of the source document")
    quote: str = Field(description="Exact quote from the document supporting the claim")

class RAGResponse(BaseModel):
    answer: str = Field(description="The generated answer based on the context")
    citations: List[Citation] = Field(description="List of citations supporting the answer")
    
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    hybrid: bool = True
