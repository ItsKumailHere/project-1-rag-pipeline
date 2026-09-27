# RAG Pipeline Architecture

This pipeline follows a strict layered architecture:
- **API Layer** (`src/api`): Handles HTTP routing and validation.
- **Service Layer** (`src/services`): Contains the business logic (Retrieval, Generation).
- **Repository Layer** (`src/repositories`): Handles all database interactions (pgvector).

## Key Patterns
- **Hybrid Retrieval**: Combines sparse (BM25) and dense (Vector) retrieval using Reciprocal Rank Fusion (RRF).
- **Structured Outputs**: All LLM outputs are validated using Pydantic models.
- **Swappable LLM Interface**: Utilizes the OpenAI SDK but allows changing `base_url` to support Groq, Gemini, or local models.
