"""
Basic retrieval pipeline combining vector and BM25 search.
"""

import logging
from typing import List, Tuple
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class BasicRetriever:
    """Retrieves relevant documents using vector similarity search."""

    def __init__(self, vector_store: VectorStore):
        """
        Initialize BasicRetriever.
        
        Args:
            vector_store: VectorStore instance for semantic search
        """
        self.vector_store = vector_store
        self.bm25_index = None
        self.documents = []
        
        logger.info("Initialized BasicRetriever")

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve documents using vector similarity.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of relevant Document objects
        """
        logger.debug(f"Retrieving {k} documents for query: {query[:50]}...")
        documents = self.vector_store.search(query, k=k)
        return documents

    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with similarity scores.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of (Document, score) tuples
        """
        return self.vector_store.search_with_scores(query, k=k)

    def build_bm25_index(self, documents: List[Document]) -> None:
        """
        Build BM25 index for keyword search.
        
        Args:
            documents: Documents to index
        """
        self.documents = documents
        tokenized_docs = [
            doc.page_content.lower().split() 
            for doc in documents
        ]
        self.bm25_index = BM25Okapi(tokenized_docs)
        logger.info(f"Built BM25 index for {len(documents)} documents")

    def retrieve_bm25(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Retrieve documents using BM25 keyword search.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of (Document, score) tuples
        """
        if self.bm25_index is None:
            logger.warning("BM25 index not built. Call build_bm25_index first.")
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top k
        top_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:k]
        
        results = [
            (self.documents[i], scores[i]) 
            for i in top_indices
        ]
        
        return results

    def retrieve_hybrid(
        self, 
        query: str, 
        k: int = 5,
        alpha: float = 0.5
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve using hybrid search (vector + BM25).
        
        Args:
            query: Search query
            k: Number of documents to return
            alpha: Weight for vector search (1-alpha for BM25)
                   alpha=1.0 means pure vector search
                   alpha=0.0 means pure BM25
            
        Returns:
            List of (Document, score) tuples
        """
        if self.bm25_index is None:
            logger.warning("BM25 index not built. Using vector search only.")
            return self.retrieve_with_scores(query, k)
        
        # Vector search results
        vector_results = self.retrieve_with_scores(query, k*2)
        vector_dict = {doc.metadata.get("chunk_id"): score for doc, score in vector_results}
        
        # BM25 results
        bm25_results = self.retrieve_bm25(query, k*2)
        bm25_dict = {doc.metadata.get("chunk_id"): score for doc, score in bm25_results}
        
        # Normalize and combine scores
        all_doc_ids = set(vector_dict.keys()) | set(bm25_dict.keys())
        combined_scores = {}
        
        for doc_id in all_doc_ids:
            vector_score = vector_dict.get(doc_id, 0)
            bm25_score = bm25_dict.get(doc_id, 0)
            
            # Normalize BM25 score (typically 0-3)
            normalized_bm25 = min(bm25_score / 3.0, 1.0)
            
            combined = (alpha * vector_score) + ((1 - alpha) * normalized_bm25)
            combined_scores[doc_id] = combined
        
        # Get top k
        top_docs = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        # Convert back to Document objects
        results = []
        for doc_id, score in top_docs:
            doc = vector_dict.get(doc_id) or bm25_dict.get(doc_id)
            if doc:
                results.append((doc, score))
        
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Retrieval Pipeline Module")
    print("=" * 60)
    print("Methods:")
    print("  - retrieve(): Vector similarity search")
    print("  - retrieve_bm25(): Keyword search")
    print("  - retrieve_hybrid(): Combined vector + keyword")
