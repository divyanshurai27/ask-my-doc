"""
End-to-end RAG pipeline integration.
Comprehensive demo showing full document ingestion → chunking → embedding → retrieval → answer generation flow.
"""

import logging
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Settings
from src.ingestion.loaders import DocumentLoaders
from src.ingestion.chunking import DocumentChunker
from src.storage.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import ReRanker
from src.rag.answer_generator import RAGChain
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline orchestrating all components."""

    def __init__(self, config: Settings = None):
        """
        Initialize RAG pipeline.
        
        Args:
            config: Settings configuration (uses defaults if None)
        """
        self.config = config or Settings()
        
        # Initialize components
        self.loaders = DocumentLoaders()
        self.chunker = DocumentChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.vector_store = VectorStore(
            persist_dir=self.config.vector_store_path,
            embedding_model=self.config.embedding_model
        )
        self.retriever = HybridRetriever(self.vector_store)
        self.reranker = ReRanker()
        self.rag_chain = RAGChain(model_name=self.config.llm_model)
        
        self.loaded_documents = []
        self.chunked_documents = []
        
        logger.info("Initialized RAG Pipeline")

    def ingest_documents(self, directory: str) -> int:
        """
        Ingest documents from directory.
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            Number of documents loaded
        """
        logger.info(f"Ingesting documents from {directory}")
        
        # Load documents
        path = Path(directory)
        file_paths = []
        if path.is_dir():
            for f in path.glob("**/*"):
                if f.is_file() and f.suffix.lower() in ['.pdf', '.md', '.markdown', '.txt']:
                    file_paths.append(str(f))
        else:
            file_paths = [directory]
            
        self.loaded_documents = self.loaders.load_documents(file_paths)
        logger.info(f"Loaded {len(self.loaded_documents)} documents")
        
        # Chunk documents
        self.chunked_documents = self.chunker.chunk_documents(self.loaded_documents)
        logger.info(f"Created {len(self.chunked_documents)} chunks")
        
        # Add to vector store
        added = self.vector_store.add_documents(self.chunked_documents)
        logger.info(f"Added {added} chunks to vector store")
        
        # Build BM25 index for hybrid search
        self.retriever.build_bm25_index(self.chunked_documents)
        
        return len(self.loaded_documents)

    def query(
        self,
        question: str,
        top_k: int = 5,
        retrieval_method: str = "vector"
    ) -> dict:
        """
        Query the RAG pipeline.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            retrieval_method: "vector", "bm25", or "hybrid"
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing query: {question[:50]}...")
        
        # Retrieve documents
        if retrieval_method == "vector":
            retrieved_scores = self.retriever.retrieve_vector(question, k=top_k * 2)
        elif retrieval_method == "bm25":
            retrieved_scores = self.retriever.retrieve_bm25(question, k=top_k * 2)
        elif retrieval_method == "hybrid":
            retrieved_scores = self.retriever.retrieve_rrf(question, k=top_k * 2)
        else:
            retrieved_scores = self.retriever.retrieve_vector(question, k=top_k * 2)
            
        # Extract just documents
        docs_to_rerank = [doc for doc, _ in retrieved_scores]
        
        # Re-rank using Cross-Encoder
        if docs_to_rerank:
            reranked = self.reranker.rerank(question, docs_to_rerank, top_k=top_k)
            retrieved = [doc for doc, _ in reranked]
        else:
            retrieved = []
        
        logger.info(f"Retrieved and re-ranked to {len(retrieved)} documents")
        
        # Generate answer
        result = self.rag_chain.generate_answer(question, retrieved)
        
        # Validate
        validation = self.rag_chain.validate_answer(result["answer"], retrieved)
        
        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "validation": validation,
            "retrieval_method": retrieval_method,
            "documents_retrieved": len(retrieved),
        }

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "loaded_documents": len(self.loaded_documents),
            "chunked_documents": len(self.chunked_documents),
            "vector_store": self.vector_store.get_stats(),
        }


def demo_basic_usage():
    """Demonstrate basic usage with sample documents."""
    logger.info("=" * 70)
    logger.info("PHASE 1 RAG PIPELINE DEMO")
    logger.info("=" * 70)
    
    # Create pipeline
    config = Settings()
    pipeline = RAGPipeline(config)
    
    # Check if sample documents exist
    sample_dir = Path("./data/sample_docs")
    if not sample_dir.exists():
        logger.error(f"Sample documents directory not found: {sample_dir}")
        logger.info("Please create sample documents in ./data/sample_docs/")
        logger.info("Supported formats: .pdf, .md, .txt, .html")
        return
    
    # Ingest documents
    logger.info("\n1. INGESTION PHASE")
    logger.info("-" * 70)
    num_docs = pipeline.ingest_documents(str(sample_dir))
    
    if num_docs == 0:
        logger.warning("No documents found. Please add sample documents.")
        return
    
    # Show statistics
    logger.info("\n2. STATISTICS")
    logger.info("-" * 70)
    stats = pipeline.get_stats()
    logger.info(f"Documents loaded: {stats['loaded_documents']}")
    logger.info(f"Chunks created: {stats['chunked_documents']}")
    logger.info(f"Vector store: {stats['vector_store']}")
    
    # Show chunk statistics
    logger.info("\n3. CHUNK ANALYSIS")
    logger.info("-" * 70)
    chunk_stats = pipeline.chunker.get_chunk_stats(pipeline.chunked_documents)
    logger.info(f"Average chunk size: {chunk_stats['avg_chunk_size']:.0f} characters")
    logger.info(f"Average chunk size: {chunk_stats['avg_chunk_size_words']:.0f} words")
    
    # Example queries
    logger.info("\n4. EXAMPLE QUERIES")
    logger.info("-" * 70)
    
    # These are example queries - they'll only work if sample docs contain relevant content
    sample_queries = [
        "What is the main topic of these documents?",
        "Can you summarize the key points?",
        "What important information is mentioned?",
    ]
    
    for i, query in enumerate(sample_queries[:1], 1):  # Just show 1 example
        logger.info(f"\nQuery {i}: {query}")
        logger.info("-" * 70)
        
        try:
            result = pipeline.query(query, top_k=3, retrieval_method="vector")
            
            logger.info(f"\nANSWER:\n{result['answer'][:500]}...")
            logger.info(f"\nSOURCES ({len(result['sources'])}):")
            for source in result['sources']:
                logger.info(f"  [{source['index']}] {source['source']}")
            
            logger.info(f"\nVALIDATION:")
            for key, value in result['validation'].items():
                logger.info(f"  {key}: {value}")
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.info("Tip: Ensure OpenAI API key is set in .env file")
    
    logger.info("\n" + "=" * 70)
    logger.info("INTERACTIVE CLI MODE")
    logger.info("=" * 70)
    logger.info("Type 'exit' or 'quit' to close the program.")
    
    while True:
        try:
            user_query = input("\nAsk a question: ").strip()
            if user_query.lower() in ['exit', 'quit']:
                break
            if not user_query:
                continue
                
            logger.info(f"\nProcessing query...")
            result = pipeline.query(user_query, top_k=3, retrieval_method="hybrid")
            
            logger.info(f"\nANSWER:\n{result['answer']}")
            logger.info(f"\nSOURCES ({len(result['sources'])}):")
            for source in result['sources']:
                logger.info(f"  [{source['index']}] {source['source']}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")

    logger.info("\n" + "=" * 70)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    demo_basic_usage()
