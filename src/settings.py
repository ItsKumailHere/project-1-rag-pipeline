from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Model Provider (OpenRouter/Groq/Gemini via OpenAI SDK)
    llm_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "google/gemini-pro-1.5"
    
    # Embeddings (Gemini or local)
    embedding_api_key: str = ""
    embedding_base_url: str = "https://openrouter.ai/api/v1" 
    embedding_model: str = "google/gemini-embedding-001"
    
    # Database (Neon preferred for simplicity with pgvector)
    database_url: str = "postgresql://user:pass@host:5432/dbname"
    
    # Evaluation (RAGAS requires OpenAI compatible)
    eval_api_key: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
