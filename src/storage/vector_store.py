"""
ChromaDB vector store integration.
Handles document embeddings and semantic search.
"""

import logging
from typing import List, Tuple
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages document embeddings and vector similarity search using ChromaDB."""

    def __init__(self, persist_dir: str = "./data/chroma_db", embedding_model: str = None):
        """
        Initialize VectorStore.
        
        Args:
            persist_dir: Directory to persist ChromaDB data
            embedding_model: Model name for embeddings (defaults to all-MiniLM-L6-v2)
        """
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized VectorStore with ChromaDB at {persist_dir}")

    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to vector store.
        
        Args:
            documents: List of Document objects with page_content and metadata
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        # Extract content and embeddings
        contents = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.encode(contents, show_progress_bar=True)
        
        # Prepare metadata (convert non-string values to strings for ChromaDB)
        metadatas = []
        for doc in documents:
            meta = {
                str(k): str(v) for k, v in doc.metadata.items()
            }
            metadatas.append(meta)
        
        # Generate IDs
        ids = [doc.metadata.get("chunk_id", f"doc_{i}") for i, doc in enumerate(documents)]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=contents,
            metadatas=metadatas,
        )
        
        logger.info(f"Added {len(documents)} documents to vector store")
        return len(documents)

    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of Document objects with metadata
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )
        
        # Convert results to Document objects
        documents = []
        if results and results["documents"] and len(results["documents"]) > 0:
            for i, content in enumerate(results["documents"][0]):
                # Get metadata and distances
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                
                # Convert distance to similarity score (cosine)
                similarity = 1 - distance
                
                doc = Document(
                    page_content=content,
                    metadata={**metadata, "similarity_score": similarity}
                )
                documents.append(doc)
        
        return documents

    def search_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search with similarity scores.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        documents = self.search(query, k)
        return [
            (doc, doc.metadata.get("similarity_score", 0.0))
            for doc in documents
        ]

    def delete_collection(self) -> None:
        """Delete all documents from collection."""
        self.client.delete_collection("documents")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Cleared vector store collection")

    def get_stats(self) -> dict:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection stats
        """
        count = self.collection.count()
        return {
            "total_documents": count,
            "embedding_model": self.embedding_model_name,
            "persist_directory": self.persist_dir,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Vector Store Module")
    print("=" * 60)
    print("Features:")
    print("  - ChromaDB with persistent storage")
    print("  - Sentence-Transformers embeddings (local, free)")
    print("  - Cosine similarity search")
    print("  - Metadata preservation")
