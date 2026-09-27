# Production RAG Pipeline

This repository demonstrates a production-grade Retrieval-Augmented Generation (RAG) pipeline built with Python, FastAPI, and PostgreSQL (pgvector). 

It specifically implements architectural patterns championed by AI engineering experts (like Dave Ebbelaar), focusing on:
1. **Layered Architecture**: Clean separation between API, Services, and Repositories.
2. **Hybrid Retrieval**: Combining dense vector search with sparse search (via Reciprocal Rank Fusion) to handle both semantic queries and exact-keyword matches.
3. **Structured Outputs**: Using Pydantic models to guarantee the LLM returns verifiable citations alongside its answer.
4. **Provider Agnosticism**: Built on the OpenAI SDK, but configured via `.env` to swap instantly to OpenRouter, Groq, or local models.

## Setup
1. `uv sync` (or `pip install -e .`)
2. Copy `.env.example` to `.env` and fill in your keys (OpenRouter/Groq/OpenAI) and Neon Postgres URL.
3. Run with `uvicorn src.api.app:app --reload`

## Evaluation
This project includes a synthetic "Golden Dataset" and a RAGAS evaluation harness to quantitatively measure Faithfulness, Answer Relevance, and Context Precision.
