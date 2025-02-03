"""Document loading and text embedding utilities."""
from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

from .config import settings


class DocumentLoader:
    """Handles PDF loading and chunking."""
    
    def __init__(self):
        self.splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
    
    def load_and_chunk_pdf(self, path: str) -> list[str]:
        """Load PDF and split into chunks."""
        docs = PDFReader().load_data(file=path)
        texts = [d.text for d in docs if getattr(d, "text", None)]
        chunks = []
        for t in texts:
            chunks.extend(self.splitter.split_text(t))
        return chunks


class EmbeddingService:
    """Handles text embeddings using OpenAI."""
    
    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set for embeddings")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]
