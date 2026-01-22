"""
Embedding and Vector Store module.

STEP 3: Create embeddings and store in vector database.
Uses sentence-transformers for embeddings and FAISS for vector search.
"""
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from config import EMBEDDING_MODEL, VECTOR_STORE_DIR
from ingest import CodeChunk


class VectorStore:
    """
    Vector store using FAISS for efficient similarity search.
    Stores embeddings and their corresponding chunks.
    """
    
    def __init__(self, store_dir: Optional[Path] = None):
        self.store_dir = store_dir or VECTOR_STORE_DIR
        self.store_dir = Path(self.store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.index = None
        self.chunks: list[CodeChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        
    def _load_model(self):
        """Lazy-load the embedding model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"🔄 Loading embedding model: {EMBEDDING_MODEL}")
                self.model = SentenceTransformer(EMBEDDING_MODEL)
                print("✅ Model loaded")
            except ImportError:
                raise ImportError(
                    "Please install sentence-transformers: pip install sentence-transformers"
                )
        return self.model
    
    def _init_faiss_index(self, dimension: int):
        """Initialize FAISS index."""
        try:
            import faiss
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity on normalized vectors)
        except ImportError:
            print("⚠️ FAISS not available, using numpy-based search")
            self.index = None
    
    def embed_chunks(self, chunks: list[CodeChunk]) -> np.ndarray:
        """
        Create embeddings for all chunks.
        
        Args:
            chunks: List of CodeChunk objects
            
        Returns:
            Numpy array of embeddings
        """
        model = self._load_model()
        self.chunks = chunks
        
        # Prepare texts for embedding
        texts = []
        for chunk in chunks:
            # Include file path for better context
            text = f"File: {chunk.file_path}\n{chunk.content}"
            texts.append(text)
        
        print(f"🔄 Embedding {len(texts)} chunks...")
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        self.embeddings = embeddings
        
        # Initialize FAISS index
        self._init_faiss_index(embeddings.shape[1])
        if self.index is not None:
            import faiss
            self.index.add(embeddings.astype(np.float32))
        
        print(f"✅ Embedded {len(chunks)} chunks (dim={embeddings.shape[1]})")
        return embeddings
    
    def search(self, query: str, top_k: int = 10) -> list[tuple[CodeChunk, float]]:
        """
        Search for chunks similar to query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples, sorted by relevance
        """
        if self.embeddings is None or len(self.chunks) == 0:
            raise ValueError("No embeddings loaded. Run embed_chunks first or load from disk.")
        
        model = self._load_model()
        
        # Embed query
        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        
        if self.index is not None:
            # Use FAISS for search
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
            results = [(self.chunks[idx], float(score)) for idx, score in zip(indices[0], scores[0])]
        else:
            # Fallback: numpy cosine similarity
            similarities = np.dot(self.embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            results = [(self.chunks[idx], float(similarities[idx])) for idx in top_indices]
        
        return results
    
    def save(self, name: str = "default"):
        """Save embeddings and chunks to disk."""
        save_path = self.store_dir / name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(save_path / "embeddings.npy", self.embeddings)
        
        # Save chunks metadata
        chunks_data = [
            {
                "file_path": c.file_path,
                "content": c.content,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "chunk_type": c.chunk_type
            }
            for c in self.chunks
        ]
        with open(save_path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)
        
        # Save FAISS index if available
        if self.index is not None:
            import faiss
            faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        print(f"💾 Saved vector store to {save_path}")
    
    def load(self, name: str = "default") -> bool:
        """Load embeddings and chunks from disk."""
        load_path = self.store_dir / name
        
        if not load_path.exists():
            print(f"⚠️ No saved store found at {load_path}")
            return False
        
        # Load embeddings
        embeddings_path = load_path / "embeddings.npy"
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
        
        # Load chunks
        chunks_path = load_path / "chunks.json"
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            self.chunks = [
                CodeChunk(
                    file_path=c["file_path"],
                    content=c["content"],
                    start_line=c["start_line"],
                    end_line=c["end_line"],
                    chunk_type=c["chunk_type"]
                )
                for c in chunks_data
            ]
        
        # Load FAISS index if available
        index_path = load_path / "index.faiss"
        if index_path.exists():
            try:
                import faiss
                self.index = faiss.read_index(str(index_path))
            except ImportError:
                print("⚠️ FAISS not available, using numpy search")
                self._init_faiss_index(self.embeddings.shape[1] if self.embeddings is not None else 384)
        
        print(f"📂 Loaded {len(self.chunks)} chunks from {load_path}")
        return True


def index_repository(repo_path: str, store_name: str = "default") -> VectorStore:
    """
    Convenience function: ingest and embed a repository.
    
    Args:
        repo_path: Path to repository
        store_name: Name for the vector store
        
    Returns:
        Populated VectorStore instance
    """
    from ingest import ingest_repository
    
    # Ingest repository
    chunks = ingest_repository(repo_path)
    
    # Create and populate vector store
    store = VectorStore()
    store.embed_chunks(chunks)
    store.save(store_name)
    
    return store


if __name__ == "__main__":
    import sys
    
    # Test: index a repository
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    store = index_repository(path, "test_index")
    
    # Test search
    query = "function that handles authentication"
    print(f"\n🔍 Searching for: '{query}'")
    results = store.search(query, top_k=3)
    
    for chunk, score in results:
        print(f"\n📄 {chunk.file_path}:{chunk.start_line}-{chunk.end_line} (score: {score:.3f})")
        print(f"   {chunk.content[:150]}...")
