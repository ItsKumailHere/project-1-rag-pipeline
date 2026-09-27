from openai import OpenAI
from .settings import settings

def get_llm_client() -> OpenAI:
    """Returns an OpenAI client configured for the primary LLM provider."""
    return OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )

def get_embedding_client() -> OpenAI:
    """Returns an OpenAI client configured for the embedding model provider."""
    return OpenAI(
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
    )
