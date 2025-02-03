"""Vector database operations using Qdrant."""
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from .config import settings


class VectorStore:
    """Manages vector storage and retrieval using Qdrant."""
    
    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        dim: int | None = None
    ):
        self.url = url or settings.qdrant_url
        self.collection = collection or settings.qdrant_collection
        self.dim = dim or settings.embedding_dim
        self._client = None
    
    @property
    def client(self) -> QdrantClient:
        """Lazy-load Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(url=self.url, timeout=30)
            self._ensure_collection()
        return self._client
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
    
    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]):
        """Insert or update vectors in the collection."""
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(self.collection, points=points)
    
    def search(self, query_vector: list[float], top_k: int = 5) -> dict[str, list]:
        """Search for similar vectors."""
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k
        )
        
        contexts = []
        sources = set()
        
        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)
        
        return {"contexts": contexts, "sources": list(sources)}
