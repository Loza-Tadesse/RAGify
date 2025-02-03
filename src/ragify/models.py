"""Pydantic models for RAGify."""
from pydantic import BaseModel


class RAGChunkAndSrc(BaseModel):
    """Document chunks with source information."""
    chunks: list[str]
    source_id: str | None = None


class RAGUpsertResult(BaseModel):
    """Result of upserting documents to vector store."""
    ingested: int


class RAGSearchResult(BaseModel):
    """Search results from vector store."""
    contexts: list[str]
    sources: list[str]


class RAGQueryResult(BaseModel):
    """Query result with answer and sources."""
    answer: str
    sources: list[str]
    num_contexts: int
