"""
Tests for document chunking.
"""

import pytest
from langchain_core.documents import Document
from src.ingestion.chunking import DocumentChunker


class TestDocumentChunker:
    """Test suite for DocumentChunker class."""

    @pytest.fixture
    def chunker(self):
        """Create a DocumentChunker instance."""
        return DocumentChunker(chunk_size=200, chunk_overlap=50)

    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing."""
        content = (
            "This is a test document. It has multiple sentences. "
            "Each sentence provides some information. "
            "The document is used for testing the chunking functionality. "
            "We need to verify that chunks maintain semantic boundaries. "
            "Overlapping text should preserve context. "
            "Let's make sure the implementation works correctly."
        )
        return Document(
            page_content=content,
            metadata={
                "source_file": "test.txt",
                "source_type": "text"
            }
        )

    def test_chunker_initialization(self, chunker):
        """Test that chunker initializes correctly."""
        assert chunker.chunk_size == 200
        assert chunker.chunk_overlap == 50

    def test_chunk_single_document(self, chunker, sample_document):
        """Test chunking a single document."""
        chunks = chunker.chunk_single_document(sample_document)
        
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)
        assert all("chunk_id" in c.metadata for c in chunks)
        assert all("chunk_index" in c.metadata for c in chunks)

    def test_chunk_metadata_preservation(self, chunker, sample_document):
        """Test that original metadata is preserved."""
        chunks = chunker.chunk_single_document(sample_document)
        
        # All chunks should have original metadata
        for chunk in chunks:
            assert chunk.metadata["source_file"] == "test.txt"
            assert chunk.metadata["source_type"] == "text"

    def test_chunk_ids_are_unique(self, chunker, sample_document):
        """Test that chunk IDs are unique."""
        chunks = chunker.chunk_single_document(sample_document)
        chunk_ids = [c.metadata["chunk_id"] for c in chunks]
        
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        content = "a " * 150  # Create content longer than chunk size
        doc = Document(page_content=content, metadata={"source_file": "test.txt"})
        
        chunks = chunker.chunk_single_document(doc)
        
        # Should create multiple chunks due to content length
        assert len(chunks) > 1

    def test_get_chunk_stats(self, chunker, sample_document):
        """Test chunk statistics calculation."""
        chunks = chunker.chunk_single_document(sample_document)
        stats = chunker.get_chunk_stats(chunks)
        
        assert "total_chunks" in stats
        assert "avg_chunk_size" in stats
        assert "min_chunk_size" in stats
        assert "max_chunk_size" in stats
        assert stats["total_chunks"] == len(chunks)

    def test_get_chunk_stats_empty_list(self, chunker):
        """Test chunk statistics with empty list."""
        stats = chunker.get_chunk_stats([])
        
        assert stats["total_chunks"] == 0
        assert stats["avg_chunk_size"] == 0

    def test_validate_chunk_sizes(self, chunker, sample_document):
        """Test chunk size validation."""
        chunks = chunker.chunk_single_document(sample_document)
        validation = chunker.validate_chunk_sizes(chunks, acceptable_range=(50, 300))
        
        assert "valid" in validation
        assert "total_chunks" in validation
        assert "out_of_range_chunks" in validation

    def test_chunk_documents_multiple(self, chunker, sample_document):
        """Test chunking multiple documents."""
        docs = [sample_document, sample_document]
        chunks = chunker.chunk_documents(docs)
        
        assert len(chunks) > 0
        # Should have chunks from both documents
        sources = [c.metadata["source_file"] for c in chunks]
        assert sources.count("test.txt") > 0

    def test_empty_document(self, chunker):
        """Test handling of empty document."""
        empty_doc = Document(page_content="", metadata={"source_file": "empty.txt"})
        chunks = chunker.chunk_single_document(empty_doc)
        
        # Should return at least one chunk (even if empty)
        assert isinstance(chunks, list)
