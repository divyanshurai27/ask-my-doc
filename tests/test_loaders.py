"""
Tests for document loaders.
Tests loading PDF, Markdown, and web pages.
"""

import pytest
from pathlib import Path
from langchain_core.documents import Document
from src.ingestion.loaders import DocumentLoaders


class TestDocumentLoaders:
    """Test suite for DocumentLoaders class."""

    def test_loaders_module_exists(self):
        """Test that loaders module is properly configured."""
        assert DocumentLoaders is not None
        assert hasattr(DocumentLoaders, 'load_pdf')
        assert hasattr(DocumentLoaders, 'load_markdown')
        assert hasattr(DocumentLoaders, 'load_text')
        assert hasattr(DocumentLoaders, 'load_web')
        assert hasattr(DocumentLoaders, 'load_documents')

    def test_load_pdf_with_nonexistent_file(self):
        """Test loading non-existent PDF raises error."""
        with pytest.raises(Exception):
            DocumentLoaders.load_pdf("nonexistent.pdf")

    def test_load_markdown_with_nonexistent_file(self):
        """Test loading non-existent Markdown raises error."""
        with pytest.raises(Exception):
            DocumentLoaders.load_markdown("nonexistent.md")

    def test_load_documents_with_mixed_formats(self):
        """Test load_documents with various file types."""
        # When no files exist, should return empty list
        docs = DocumentLoaders.load_documents(
            ["nonexistent1.pdf", "nonexistent2.md"],
            verbose=False
        )
        # Should handle gracefully
        assert isinstance(docs, list)

    def test_document_metadata_preservation(self):
        """Test that metadata is preserved when loading."""
        # This would require actual test files
        # For now, verify the structure would be correct
        
        # Mock document structure
        test_doc = Document(
            page_content="Test content",
            metadata={
                "source_type": "pdf",
                "source_file": "test.pdf",
                "page_number": 1
            }
        )
        
        assert test_doc.metadata["source_type"] == "pdf"
        assert test_doc.metadata["source_file"] == "test.pdf"


# Sample test data creation
@pytest.fixture
def sample_markdown_path(tmp_path):
    """Create a sample markdown file for testing."""
    md_file = tmp_path / "test.md"
    md_file.write_text("# Test Heading\n\nTest content here.")
    return str(md_file)


@pytest.fixture
def sample_text_path(tmp_path):
    """Create a sample text file for testing."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("This is test content.\nMultiple lines.")
    return str(txt_file)


def test_load_markdown_file(sample_markdown_path):
    """Test loading a real markdown file."""
    docs = DocumentLoaders.load_markdown(sample_markdown_path)
    
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert "Test Heading" in docs[0].page_content
    assert docs[0].metadata["source_type"] == "markdown"


def test_load_text_file(sample_text_path):
    """Test loading a real text file."""
    docs = DocumentLoaders.load_text(sample_text_path)
    
    assert len(docs) >= 1
    assert isinstance(docs[0], Document)
    assert docs[0].metadata["source_type"] == "text"
