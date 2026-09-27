import logging
from typing import List
from src.repositories.document_repo import DocumentRepository
from src.clients import get_embedding_client
from src.settings import settings
from src.models.schemas import SearchResult
from src.utils.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self):
        self.repo = DocumentRepository()
        self.emb_client = get_embedding_client()
        
    def _get_embedding(self, text: str) -> List[float]:
        response = self.emb_client.embeddings.create(
            input=[text],
            model=settings.embedding_model
        )
        return response.data[0].embedding
        
    def search(self, query: str, top_k: int = 5, hybrid: bool = True) -> List[SearchResult]:
        """
        Executes search. Currently supports dense vector search.
        In a full implementation, BM25 sparse search would be run in parallel
        and the results fused via RRF.
        """
        logger.info(f"Embedding query using {settings.embedding_model}")
        query_emb = self._get_embedding(query)
        
        # Dense Retrieval (pgvector)
        logger.info(f"Executing vector search for top {top_k} results")
        # In a real hybrid setup we might fetch candidate_k=50 for both, then fuse, then rerank
        dense_results = self.repo.vector_search(query_emb, top_k=top_k * 2) 
        
        if not hybrid:
            return dense_results[:top_k]
            
        # Mocking hybrid fusion for now since we don't have a persistent BM25 index setup yet
        # In full implementation:
        # bm25_results = self.bm25_search(query, top_k=top_k * 2)
        # return reciprocal_rank_fusion([bm25_results, dense_results], top_n=top_k)
        
        # We just return the dense results for now to keep the API contract
        return dense_results[:top_k]
