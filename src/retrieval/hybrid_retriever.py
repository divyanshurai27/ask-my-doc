"""
Advanced hybrid retrieval with Reciprocal Rank Fusion.
Combines BM25 keyword search and vector semantic search optimally.
"""

import logging
from typing import List, Tuple, Dict
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Advanced hybrid retrieval combining BM25 and vector search with RRF fusion."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        """
        Initialize HybridRetriever.
        
        Args:
            vector_store: VectorStore instance
            bm25_k1: BM25 parameter (saturation parameter, default 1.5)
            bm25_b: BM25 parameter (length normalization, default 0.75)
        """
        self.vector_store = vector_store
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.bm25_index = None
        self.documents = []
        
        logger.info(f"Initialized HybridRetriever (k1={bm25_k1}, b={bm25_b})")

    def build_bm25_index(self, documents: List[Document]) -> None:
        """
        Build BM25 index with tuned parameters.
        
        Args:
            documents: Documents to index
        """
        self.documents = documents
        tokenized_docs = [
            doc.page_content.lower().split() 
            for doc in documents
        ]
        
        # Initialize BM25 with tuned parameters
        self.bm25_index = BM25Okapi(
            tokenized_docs,
            k1=self.bm25_k1,
            b=self.bm25_b
        )
        logger.info(f"Built BM25 index for {len(documents)} documents")

    def retrieve_bm25(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Retrieve using BM25 with ranking.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (Document, score) tuples, ranked by BM25
        """
        if self.bm25_index is None:
            logger.warning("BM25 index not built")
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
            (self.documents[i], float(scores[i]))
            for i in top_indices
        ]
        
        return results

    def retrieve_vector(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Retrieve using vector similarity.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (Document, score) tuples
        """
        return self.vector_store.search_with_scores(query, k=k)

    def retrieve_rrf(
        self,
        query: str,
        k: int = 5,
        rrf_constant: int = 60
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve using Reciprocal Rank Fusion (RRF).
        
        RRF formula: score = 1/(rrf_constant + rank)
        Combines rankings from both BM25 and vector search.
        
        Args:
            query: Search query
            k: Number of results to return
            rrf_constant: Constant for RRF (typically 60)
            
        Returns:
            List of (Document, score) tuples
        """
        if self.bm25_index is None:
            logger.warning("BM25 index not built, using vector search only")
            return self.retrieve_vector(query, k)
        
        # Get results from both methods (fetch more to ensure good combination)
        bm25_results = self.retrieve_bm25(query, k=k*2)
        vector_results = self.retrieve_vector(query, k=k*2)
        
        # Create ranking dictionaries with RRF scores
        rrf_scores = {}
        
        # BM25 rankings
        for rank, (doc, _) in enumerate(bm25_results, 1):
            doc_id = doc.metadata.get("chunk_id")
            rrf_score = 1.0 / (rrf_constant + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score
        
        # Vector rankings
        for rank, (doc, _) in enumerate(vector_results, 1):
            doc_id = doc.metadata.get("chunk_id")
            rrf_score = 1.0 / (rrf_constant + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score
        
        # Sort by combined RRF scores
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        # Convert back to Document objects
        all_docs = {doc.metadata.get("chunk_id"): doc for doc, _ in bm25_results + vector_results}
        results = [
            (all_docs[doc_id], score)
            for doc_id, score in sorted_docs
            if doc_id in all_docs
        ]
        
        logger.debug(f"RRF retrieved {len(results)} documents for query: {query[:50]}...")
        return results

    def retrieve_weighted(
        self,
        query: str,
        k: int = 5,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve using weighted combination of scores.
        
        Args:
            query: Search query
            k: Number of results
            vector_weight: Weight for vector search scores (0.0-1.0)
            bm25_weight: Weight for BM25 scores (0.0-1.0)
            
        Returns:
            List of (Document, score) tuples
        """
        if abs((vector_weight + bm25_weight) - 1.0) > 0.01:
            logger.warning(f"Weights should sum to 1.0, got {vector_weight + bm25_weight}")
        
        if self.bm25_index is None:
            logger.warning("BM25 index not built, using vector search only")
            return self.retrieve_vector(query, k)
        
        # Get results from both
        bm25_results = self.retrieve_bm25(query, k=k*2)
        vector_results = self.retrieve_vector(query, k=k*2)
        
        # Normalize scores and combine
        weighted_scores = {}
        
        # BM25 (normalize to 0-1 range)
        if bm25_results:
            max_bm25_score = max(score for _, score in bm25_results)
            for doc, score in bm25_results:
                doc_id = doc.metadata.get("chunk_id")
                normalized = score / max_bm25_score if max_bm25_score > 0 else 0
                weighted_scores[doc_id] = bm25_weight * normalized
        
        # Vector (already 0-1 range)
        if vector_results:
            for doc, score in vector_results:
                doc_id = doc.metadata.get("chunk_id")
                weighted_scores[doc_id] = weighted_scores.get(doc_id, 0) + (vector_weight * score)
        
        # Sort and return top k
        sorted_docs = sorted(
            weighted_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        all_docs = {doc.metadata.get("chunk_id"): doc for doc, _ in bm25_results + vector_results}
        results = [
            (all_docs[doc_id], score)
            for doc_id, score in sorted_docs
            if doc_id in all_docs
        ]
        
        return results

    def get_retrieval_stats(self) -> Dict[str, any]:
        """Get retrieval statistics."""
        return {
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "bm25_indexed_documents": len(self.documents),
            "vector_store": self.vector_store.get_stats(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Advanced Hybrid Retrieval Module")
    print("=" * 60)
    print("Methods:")
    print("  - retrieve_rrf(): Reciprocal Rank Fusion")
    print("  - retrieve_weighted(): Weighted score combination")
    print("  - retrieve_bm25(): Pure keyword search")
    print("  - retrieve_vector(): Pure semantic search")
