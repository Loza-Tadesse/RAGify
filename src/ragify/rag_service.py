"""RAG service orchestrating document processing and querying."""
import uuid
from typing import Protocol

from .config import settings
from .models import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult
from .document_processor import DocumentLoader, EmbeddingService
from .vector_store import VectorStore


class LLMAdapter(Protocol):
    """Protocol for LLM adapters."""
    async def infer(self, messages: list[dict], **kwargs) -> dict:
        """Generate inference from LLM."""
        ...


class RAGService:
    """Orchestrates RAG operations."""
    
    def __init__(
        self,
        document_loader: DocumentLoader | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None
    ):
        self.document_loader = document_loader or DocumentLoader()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()
    
    def ingest_document(self, pdf_path: str, source_id: str | None = None) -> RAGUpsertResult:
        """Load, chunk, embed, and store a PDF document."""
        source_id = source_id or pdf_path
        
        # Load and chunk
        chunks = self.document_loader.load_and_chunk_pdf(pdf_path)
        
        # Generate embeddings
        vectors = self.embedding_service.embed_texts(chunks)
        
        # Create IDs and payloads
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]
        payloads = [
            {"source": source_id, "text": chunks[i]}
            for i in range(len(chunks))
        ]
        
        # Upsert to vector store
        self.vector_store.upsert(ids, vectors, payloads)
        
        return RAGUpsertResult(ingested=len(chunks))
    
    def search_context(self, question: str, top_k: int = 5) -> RAGSearchResult:
        """Search for relevant context given a question."""
        # Embed the question
        query_vector = self.embedding_service.embed_texts([question])[0]
        
        # Search vector store
        found = self.vector_store.search(query_vector, top_k)
        
        return RAGSearchResult(
            contexts=found["contexts"],
            sources=found["sources"]
        )
    
    @staticmethod
    def build_prompt(question: str, contexts: list[str]) -> str:
        """Build a prompt for the LLM given question and contexts."""
        context_block = "\n\n".join(f"- {c}" for c in contexts)
        return (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer concisely using the context above."
        )
