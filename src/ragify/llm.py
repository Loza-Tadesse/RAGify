"""LLM adapter utilities for Claude and OpenAI."""
import os
from inngest.experimental import ai

from .config import settings


def get_llm_adapter() -> ai.anthropic.Adapter | ai.openai.Adapter:
    """Get the configured LLM adapter (Anthropic preferred, OpenAI fallback)."""
    anthropic_key = settings.anthropic_api_key
    
    if anthropic_key:
        # Use Anthropic Claude Sonnet when available
        return ai.anthropic.Adapter(
            auth_key=anthropic_key,
            model=settings.anthropic_model,
        )
    else:
        # Fallback to OpenAI
        if not settings.openai_api_key:
            raise ValueError("Either ANTHROPIC_API_KEY or OPENAI_API_KEY must be set")
        
        return ai.openai.Adapter(
            auth_key=settings.openai_api_key,
            model=settings.openai_model,
        )
