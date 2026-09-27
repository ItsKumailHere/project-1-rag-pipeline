import logging
from typing import List, Dict, Any, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from src.settings import settings
from src.models.schemas import DocumentChunk, SearchResult

logger = logging.getLogger(__name__)

class DocumentRepository:
    """Handles all database interactions for documents and vectors."""
    
    def __init__(self):
        # The connection string comes from settings
        self.engine = create_engine(settings.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._init_db()
        
    def _init_db(self):
        """Initializes the pgvector extension and the documents table."""
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            # Assuming embedding dim of 768 for gemini-embedding-001, 
            # adjust if using a different model.
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS documents (
                    id VARCHAR(255) PRIMARY KEY,
                    document_id VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector(768)
                )
            """))
            # Create an HNSW index for faster similarity search
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS doc_embedding_idx 
                ON documents USING hnsw (embedding vector_cosine_ops)
            """))
            
    def store_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Stores document chunks along with their vector embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
            
        with self.SessionLocal() as session:
            for chunk, emb in zip(chunks, embeddings):
                # Convert embedding list to string format required by pgvector
                emb_str = f"[{','.join(map(str, emb))}]"
                
                session.execute(
                    text("""
                        INSERT INTO documents (id, document_id, content, metadata, embedding)
                        VALUES (:id, :doc_id, :content, :metadata, :embedding)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                    """),
                    {
                        "id": chunk.id,
                        "doc_id": chunk.document_id,
                        "content": chunk.content,
                        import json
                        "metadata": json.dumps(chunk.metadata),
                        "embedding": emb_str
                    }
                )
            session.commit()
            
    def vector_search(self, query_embedding: List[float], top_k: int = 10) -> List[SearchResult]:
        """Performs a vector similarity search (cosine distance)."""
        emb_str = f"[{','.join(map(str, query_embedding))}]"
        
        with self.SessionLocal() as session:
            # We use <=> which is cosine distance in pgvector
            # Score = 1 - cosine_distance to map it to [0, 1] similarity
            results = session.execute(
                text("""
                    SELECT id, content, metadata, 1 - (embedding <=> :embedding) as score
                    FROM documents
                    ORDER BY embedding <=> :embedding
                    LIMIT :top_k
                """),
                {"embedding": emb_str, "top_k": top_k}
            )
            
            return [
                SearchResult(
                    chunk_id=row.id,
                    content=row.content,
                    metadata=row.metadata if isinstance(row.metadata, dict) else {},
                    score=float(row.score)
                )
                for row in results
            ]
