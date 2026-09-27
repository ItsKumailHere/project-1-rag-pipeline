import logging
from typing import List, Dict, Any
from src.repositories.document_repo import DocumentRepository
from src.clients import get_embedding_client
from src.settings import settings
from src.utils.chunking import simple_chunker

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self):
        self.repo = DocumentRepository()
        self.emb_client = get_embedding_client()
        
    def process_and_store(self, text: str, document_id: str, metadata: Dict[str, Any] = None):
        """Chunks text, generates embeddings, and stores in the repository."""
        logger.info(f"Processing document: {document_id}")
        
        # 1. Chunk
        chunks = simple_chunker(text, document_id, metadata=metadata)
        logger.info(f"Created {len(chunks)} chunks")
        
        # 2. Embed
        # We batch embed for efficiency
        texts = [chunk.content for chunk in chunks]
        response = self.emb_client.embeddings.create(
            input=texts,
            model=settings.embedding_model
        )
        embeddings = [data.embedding for data in response.data]
        
        # 3. Store
        self.repo.store_chunks(chunks, embeddings)
        logger.info(f"Successfully stored {len(chunks)} chunks in vector database")
