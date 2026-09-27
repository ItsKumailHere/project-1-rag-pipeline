import logging
from src.clients import get_llm_client
from src.settings import settings
from src.models.schemas import SearchResult, RAGResponse

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a highly capable technical assistant. 
Answer the user's question based ONLY on the provided context documents.
If the context does not contain the answer, explicitly state that you don't know based on the provided documents.

You must provide citations for your answer. Each citation should include the exact quote from the document that supports your claim.
"""

class GenerationService:
    def __init__(self):
        self.llm_client = get_llm_client()
        
    def generate_answer(self, query: str, context: list[SearchResult]) -> RAGResponse:
        """Generates an answer using the LLM, strictly bound to the RAGResponse Pydantic schema."""
        
        # Format the context for the prompt
        context_str = "\n\n---\n\n".join([
            f"Document ID: {res.metadata.get('document_id', res.chunk_id)}\nContent: {res.content}"
            for res in context
        ])
        
        prompt = f"Context Documents:\n{context_str}\n\nQuestion: {query}"
        
        logger.info(f"Generating answer for query using model: {settings.llm_model}")
        
        # We use the parse helper if using OpenAI native, but for OpenRouter/Gemini compatibility
        # we'll use structured outputs (JSON schema) in the standard chat completion format
        # Note: If OpenRouter model doesn't support structured outputs perfectly, we might need instructor
        
        # In this implementation, we use the instructor pattern or standard function calling
        # For simplicity and broad compatibility, we'll use OpenAI's native tools/response_format
        
        completion = self.llm_client.beta.chat.completions.parse(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format=RAGResponse,
            temperature=0.1
        )
        
        return completion.choices[0].message.parsed
