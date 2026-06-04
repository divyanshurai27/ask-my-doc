"""
Comprehensive Phase 1 integration tests.
Tests the complete RAG pipeline workflow.
"""

import pytest
import tempfile
from pathlib import Path
from langchain_core.documents import Document

from src.config import Settings
from src.ingestion.loaders import DocumentLoaders
from src.ingestion.chunking import DocumentChunker
from src.storage.vector_store import VectorStore
from src.retrieval.basic_retrieval import BasicRetriever
from src.rag.answer_generator import RAGChain


class TestPhase1Integration:
    """Integration tests for Phase 1 components."""

    @pytest.fixture
    def config(self):
        """Get configuration."""
        return Settings()

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content=(
                    "The company policy states that employees should work 8 hours per day. "
                    "Breaks are 30 minutes in the morning and 30 minutes in the afternoon. "
                    "Lunch breaks are 1 hour long. "
                    "Employees are entitled to 20 days of paid vacation per year."
                ),
                metadata={
                    "source_file": "company_policy.txt",
                    "source_type": "text",
                    "page_number": 1
                }
            ),
            Document(
                page_content=(
                    "The benefits package includes health insurance, dental coverage, and vision insurance. "
                    "All employees are automatically enrolled in the company pension plan. "
                    "The company provides a 401(k) matching program. "
                    "Annual bonuses are based on performance reviews."
                ),
                metadata={
                    "source_file": "benefits.txt",
                    "source_type": "text",
                    "page_number": 1
                }
            ),
        ]

    @pytest.fixture
    def temp_vector_store(self):
        """Create temporary vector store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(persist_dir=tmpdir)
            yield store
            # Release ChromaDB resources to unlock files on Windows
            if hasattr(store, "client") and store.client:
                if hasattr(store.client, "_system"):
                    try:
                        store.client._system.stop()
                    except Exception:
                        pass
            store.client = None
            import gc
            gc.collect()

    def test_loaders_integration(self, sample_documents):
        """Test document loaders."""
        loaders = DocumentLoaders()
        assert loaders is not None

    def test_chunking_integration(self, sample_documents):
        """Test document chunking."""
        chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk_documents(sample_documents)
        
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)
        assert all("chunk_id" in c.metadata for c in chunks)

    def test_vector_store_integration(self, temp_vector_store, sample_documents):
        """Test vector store operations."""
        chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk_documents(sample_documents)
        
        # Add documents
        added = temp_vector_store.add_documents(chunks)
        assert added == len(chunks)
        
        # Search
        results = temp_vector_store.search("insurance", k=2)
        assert len(results) <= 2
        assert all(isinstance(r, Document) for r in results)

    def test_retriever_integration(self, temp_vector_store, sample_documents):
        """Test basic retriever."""
        chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk_documents(sample_documents)
        
        # Setup
        temp_vector_store.add_documents(chunks)
        retriever = BasicRetriever(temp_vector_store)
        
        # Vector search
        docs = retriever.retrieve("insurance benefits", k=2)
        assert len(docs) <= 2
        assert all(isinstance(d, Document) for d in docs)

    def test_hybrid_retrieval(self, temp_vector_store, sample_documents):
        """Test hybrid retrieval."""
        chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk_documents(sample_documents)
        
        # Setup
        temp_vector_store.add_documents(chunks)
        retriever = BasicRetriever(temp_vector_store)
        retriever.build_bm25_index(chunks)
        
        # Hybrid search
        docs_with_scores = retriever.retrieve_hybrid("employees vacation", k=2, alpha=0.5)
        assert len(docs_with_scores) <= 2
        assert all(isinstance(d[0], Document) for d in docs_with_scores)
        assert all(isinstance(d[1], float) for d in docs_with_scores)

    def test_rag_chain_integration(self, sample_documents):
        """Test RAG chain (without actual LLM call)."""
        rag = RAGChain()
        assert rag is not None
        
        # Test citation extraction
        response = "The answer is [1] something from [2] the documents.\n\nSources:\n[1] doc1\n[2] doc2"
        answer, sources = rag._extract_citations_and_sources(response, sample_documents)
        
        assert "[" in answer
        assert len(sources) == 2

    def test_validation_integration(self, sample_documents):
        """Test answer validation."""
        rag = RAGChain()
        
        # Test valid answer
        valid_answer = "The answer is [1] supported by documents [2]."
        validation = rag.validate_answer(valid_answer, sample_documents)
        assert validation["has_citations"]
        
        # Test refusal
        refusal = "I could not find enough information in the provided documents."
        validation = rag.validate_answer(refusal, sample_documents)
        assert validation["is_refusal"]

    def test_pipeline_stats(self, temp_vector_store, sample_documents):
        """Test pipeline statistics."""
        chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk_documents(sample_documents)
        
        stats = chunker.get_chunk_stats(chunks)
        assert "total_chunks" in stats
        assert "avg_chunk_size" in stats
        assert stats["total_chunks"] > 0

    def test_chunk_validation(self, sample_documents):
        """Test chunk size validation."""
        chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk_documents(sample_documents)
        
        validation = chunker.validate_chunk_sizes(chunks, acceptable_range=(100, 500))
        assert "valid" in validation
        assert "out_of_range_chunks" in validation

    def test_metadata_preservation(self, sample_documents):
        """Test metadata preservation through pipeline."""
        chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk_documents(sample_documents)
        
        # Check metadata
        for chunk in chunks:
            assert "source_file" in chunk.metadata
            assert "source_type" in chunk.metadata
            assert "page_number" in chunk.metadata
            assert "chunk_id" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_empty_documents_handling(self):
        """Test handling of empty documents."""
        chunker = DocumentChunker()
        
        # Empty document
        empty_doc = Document(page_content="", metadata={"source_file": "empty.txt"})
        chunks = chunker.chunk_documents([empty_doc])
        
        # Should handle gracefully
        assert isinstance(chunks, list)

    def test_large_document_chunking(self):
        """Test chunking of large documents."""
        large_content = " ".join(["word"] * 1000)  # 5000 characters
        large_doc = Document(
            page_content=large_content,
            metadata={"source_file": "large.txt"}
        )
        
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_documents([large_doc])
        
        # Should split into multiple chunks
        assert len(chunks) > 1
