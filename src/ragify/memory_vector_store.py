"""In-memory vector store implementation - no external dependencies needed."""
import numpy as np
from typing import List, Dict, Tuple


class InMemoryVectorStore:
    """Simple in-memory vector storage using numpy for similarity search."""
    
    def __init__(self):
        self.vectors: List[np.ndarray] = []
        self.payloads: List[Dict] = []
        self.ids: List[str] = []
    
    def upsert(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict]):
        """Insert or update vectors."""
        for i, vec_id in enumerate(ids):
            vector = np.array(vectors[i])
            
            # Check if ID exists and update
            if vec_id in self.ids:
                idx = self.ids.index(vec_id)
                self.vectors[idx] = vector
                self.payloads[idx] = payloads[i]
            else:
                # Add new
                self.ids.append(vec_id)
                self.vectors.append(vector)
                self.payloads.append(payloads[i])
    
    def search(self, query_vector: List[float], top_k: int = 5) -> Dict[str, List]:
        """Search for most similar vectors using cosine similarity."""
        if not self.vectors:
            return {"contexts": [], "sources": []}
        
        query = np.array(query_vector)
        
        # Calculate cosine similarities
        similarities = []
        for vec in self.vectors:
            # Cosine similarity = dot product / (norm1 * norm2)
            similarity = np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec))
            similarities.append(similarity)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Extract results
        contexts = []
        sources = set()
        
        for idx in top_indices:
            payload = self.payloads[idx]
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)
        
        return {"contexts": contexts, "sources": list(sources)}
    
    def clear(self):
        """Clear all stored vectors."""
        self.vectors = []
        self.payloads = []
        self.ids = []
    
    def count(self) -> int:
        """Return number of stored vectors."""
        return len(self.vectors)
