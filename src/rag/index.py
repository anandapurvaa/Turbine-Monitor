from pathlib import Path
from typing import List, Dict, Tuple
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class ManualIndex:
    """FAISS index for maintenance manual retrieval."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.manuals: List[Dict] = []
        self.index = None
        self.embeddings = None
        
    def build_index(self, corpus_path: Path):
        """Build FAISS index from corpus JSON."""
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.manuals = json.load(f)
        
        # Create embeddings for title + content
        texts = [
            f"{m['title']}: {m['content'].strip()}"
            for m in self.manuals
        ]
        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"Built index with {len(self.manuals)} manuals")
        print(f"Embedding dimension: {dimension}")
        
    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Search for most relevant manuals."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Return manuals with scores
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.manuals):
                results.append((self.manuals[idx], float(dist)))
        
        return results
    
    def save(self, output_dir: Path):
        """Save index and manuals."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        np.save(output_dir / "embeddings.npy", self.embeddings)
        
        # Save FAISS index
        faiss.write_index(self.index, str(output_dir / "faiss.index"))
        
        # Save manuals
        with open(output_dir / "manuals.json", "w", encoding="utf-8") as f:
            json.dump(self.manuals, f, indent=2)
        
        print(f"Saved index to {output_dir}")
    
    def load(self, index_dir: Path):
        """Load index from directory."""
        # Load manuals
        with open(index_dir / "manuals.json", "r", encoding="utf-8") as f:
            self.manuals = json.load(f)
        
        # Load embeddings
        self.embeddings = np.load(index_dir / "embeddings.npy")
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        
        print(f"Loaded index with {len(self.manuals)} manuals")


def main():
    from src.rag.build_corpus import build_corpus
    
    # Build corpus
    manuals_dir = Path("data/manuals")
    build_corpus(manuals_dir)
    
    # Build index
    indexer = ManualIndex()
    indexer.build_index(manuals_dir / "corpus.json")
    
    # Save index
    indexer.save(manuals_dir / "index")
    
    # Test search
    print("\n--- Test Search ---")
    test_query = "high pressure compressor temperature rising efficiency drop"
    results = indexer.search(test_query, top_k=2)
    
    for manual, score in results:
        print(f"\nScore: {score:.3f}")
        print(f"Title: {manual['title']}")
        print(f"ID: {manual['id']}")


if __name__ == "__main__":
    main()