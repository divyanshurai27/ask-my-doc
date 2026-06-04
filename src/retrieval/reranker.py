"""
Cross-encoder based re-ranking for improved retrieval relevance.
Uses semantic cross-encoders to re-rank retrieved documents.
"""

import logging
from typing import List, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class ReRanker:
    """Re-rank documents using cross-encoder models."""

    def __init__(self, model_name: str = "cross-encoder/mmarco-MiniLMv2-L12-H384-v30"):
        """
        Initialize ReRanker.
        
        Args:
            model_name: HuggingFace model ID for cross-encoder
                       Defaults to a multilingual model optimized for speed
        """
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        logger.info("Cross-encoder model loaded")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank documents by relevance to query.
        
        Args:
            query: Query string
            documents: List of documents to re-rank
            top_k: Number of top documents to return
            
        Returns:
            List of (Document, score) tuples, sorted by relevance
        """
        if not documents:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Create ranked list
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Re-ranked {len(documents)} documents, returning top {top_k}")
        return doc_scores[:top_k]

    def batch_rerank(
        self,
        queries: List[str],
        document_lists: List[List[Document]],
        top_k: int = 5
    ) -> List[List[Tuple[Document, float]]]:
        """
        Re-rank multiple queries efficiently.
        
        Args:
            queries: List of query strings
            document_lists: List of document lists (one per query)
            top_k: Number of top documents per query
            
        Returns:
            List of ranked document lists
        """
        results = []
        for query, docs in zip(queries, document_lists):
            ranked = self.rerank(query, docs, top_k)
            results.append(ranked)
        
        return results

    def get_top_passages(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3,
        min_score: float = 0.0
    ) -> List[Document]:
        """
        Get top passages above a minimum score threshold.
        
        Args:
            query: Query string
            documents: Documents to re-rank
            top_k: Maximum number of documents
            min_score: Minimum relevance score
            
        Returns:
            List of top documents above threshold
        """
        ranked = self.rerank(query, documents, top_k=len(documents))
        
        # Filter by minimum score and limit to top_k
        filtered = [
            doc for doc, score in ranked
            if score >= min_score
        ][:top_k]
        
        return filtered

    def get_model_info(self) -> dict:
        """Get information about the re-ranker model."""
        return {
            "model_name": self.model_name,
            "model_type": "cross-encoder",
            "description": "Semantic relevance scoring for document re-ranking"
        }


class EnsembleReranker:
    """Ensemble of multiple re-rankers for robust scoring."""

    def __init__(self, models: List[str] = None):
        """
        Initialize EnsembleReranker.
        
        Args:
            models: List of model names to use in ensemble
                   Defaults to a single efficient model
        """
        if models is None:
            models = ["cross-encoder/mmarco-MiniLMv2-L12-H384-v30"]
        
        self.rerankers = [ReRanker(model) for model in models]
        logger.info(f"Initialized ensemble with {len(self.rerankers)} models")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
        aggregation: str = "mean"
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank using ensemble of models.
        
        Args:
            query: Query string
            documents: Documents to re-rank
            top_k: Number of top documents
            aggregation: How to combine scores ("mean", "max", "min")
            
        Returns:
            List of (Document, score) tuples
        """
        if not documents:
            return []
        
        # Get scores from all rerankers
        all_scores = []
        for reranker in self.rerankers:
            pairs = [[query, doc.page_content] for doc in documents]
            scores = reranker.model.predict(pairs)
            all_scores.append(scores)
        
        # Aggregate scores
        import numpy as np
        all_scores = np.array(all_scores)
        
        if aggregation == "mean":
            final_scores = np.mean(all_scores, axis=0)
        elif aggregation == "max":
            final_scores = np.max(all_scores, axis=0)
        elif aggregation == "min":
            final_scores = np.min(all_scores, axis=0)
        else:
            final_scores = np.mean(all_scores, axis=0)
        
        # Rank documents
        doc_scores = list(zip(documents, final_scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores[:top_k]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Cross-Encoder Re-ranking Module")
    print("=" * 60)
    print("Features:")
    print("  - Single-model re-ranking")
    print("  - Ensemble re-ranking with score aggregation")
    print("  - Minimum score thresholding")
    print("  - Batch re-ranking support")
