from typing import List, Dict, Any
from src.models.schemas import DocumentChunk
import uuid

def simple_chunker(text: str, document_id: str, chunk_size: int = 1000, overlap: int = 200, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
    """A naive character-based chunker for demonstration."""
    if metadata is None:
        metadata = {}
        
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append(
            DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                content=chunk_text,
                metadata=metadata
            )
        )
        start += chunk_size - overlap
        
    return chunks
