from langchain_groq import ChatGroq
from ..core.config import settings
from .vector_db import vector_db
from typing import List, Dict, Any, Generator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        # We are injecting the key directly as requested to bypass .env issues
        self.live_key = "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"
        logger.info(f"Initializing ChatGroq with model: {settings.LLM_MODEL}")
        
        self.llm = ChatGroq(
            groq_api_key=self.live_key,
            model_name=settings.LLM_MODEL,
            temperature=0.2,
            streaming=True
        )

    def stream_response(self, query: str, top_k: int = 5) -> Generator[str, None, None]:
        try:
            # 1. Retrieve context
            matching_chunks = vector_db.search(query, top_k=top_k)
            
            if not matching_chunks:
                yield "I'm sorry, I couldn't find any relevant information in the uploaded documents to answer your question."
                return

            # 2. Build context string
            context_text = "\n\n---\n\n".join([
                f"Source: {h['filename']} (Page {h['page_number']})\nContent: {h['text']}" 
                for h in matching_chunks
            ])

            # 3. Prompt
            prompt_text = f"""You are a helpful enterprise AI assistant. Use the provided context to answer the question. 
Be concise and grounded in the context. If you don't know the answer, say you don't know.

Context:
{context_text}

Question: {query}
"""

            # 4. Stream using LangChain
            for chunk in self.llm.stream(prompt_text):
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.error(f"Error in stream_response: {e}")
            yield f"Error during generation: {str(e)}"

    def generate_response(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Synchronous version for compatibility."""
        try:
            matching_chunks = vector_db.search(query, top_k=top_k)
            if not matching_chunks:
                return {"answer": "No relevant info found.", "sources": [], "context_used": False}

            context_text = "\n\n".join([f"Page {h['page_number']}: {h['text']}" for h in matching_chunks])
            prompt = f"Context:\n{context_text}\n\nQ: {query}"
            
            response = self.llm.invoke(prompt)
            return {
                "answer": response.content,
                "sources": matching_chunks,
                "context_used": True
            }
        except Exception as e:
            return {"answer": str(e), "sources": [], "context_used": False}

llm_service = LLMService()
