"""
Document chunking strategies.
Implements intelligent text splitting with configurable size and overlap.
"""

from typing import List, Dict, Any
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Split documents into chunks with configurable size and overlap."""

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        separators: List[str] = None
    ):
        """
        Initialize DocumentChunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
                       (roughly equivalent to tokens when divided by ~4)
            chunk_overlap: Number of overlapping characters between chunks
                          Preserves context across chunk boundaries
            separators: List of separators for splitting
                       Tries in order: paragraphs, newlines, sentences, words, characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            # Semantic boundaries: paragraphs -> lines -> sentences -> words -> chars
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
        
        logger.info(
            f"Initialized DocumentChunker: "
            f"size={chunk_size}, overlap={chunk_overlap}"
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects with metadata
        """
        logger.info(f"Chunking {len(documents)} documents")
        chunked_docs = []
        
        for doc in documents:
            chunks = self.chunk_single_document(doc)
            chunked_docs.extend(chunks)
        
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs

    def chunk_single_document(self, document: Document) -> List[Document]:
        """
        Chunk a single document.
        
        Args:
            document: Document object to chunk
            
        Returns:
            List of chunked Document objects
        """
        # Split document text into chunks
        chunk_texts = self.splitter.split_text(document.page_content)
        
        # Create Document objects with preserved metadata
        chunked_documents = []
        source_file = document.metadata.get('source_file', 'unknown')
        
        for i, chunk_text in enumerate(chunk_texts):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    # Preserve original metadata
                    **document.metadata,
                    # Add chunking metadata
                    "chunk_id": str(uuid.uuid4()),
                    "source_file": source_file,
                    "chunk_index": i,
                    "total_chunks": len(chunk_texts),
                    "chunk_size": len(chunk_text),
                    "chunk_size_words": len(chunk_text.split()),
                }
            )
            chunked_documents.append(chunk_doc)
        
        logger.debug(
            f"Chunked '{source_file}' into {len(chunked_documents)} chunks"
        )
        return chunked_documents

    def get_chunk_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Calculate statistics about chunks.
        
        Args:
            documents: List of chunked Document objects
            
        Returns:
            Dictionary with statistics
        """
        if not documents:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "avg_chunk_size_words": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "min_chunk_size_words": 0,
                "max_chunk_size_words": 0,
            }
        
        chunk_sizes = [len(doc.page_content) for doc in documents]
        chunk_words = [len(doc.page_content.split()) for doc in documents]
        
        return {
            "total_chunks": len(documents),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "avg_chunk_size_words": sum(chunk_words) / len(chunk_words),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "min_chunk_size_words": min(chunk_words),
            "max_chunk_size_words": max(chunk_words),
        }

    def validate_chunk_sizes(
        self,
        documents: List[Document],
        acceptable_range: tuple = (400, 800)
    ) -> Dict[str, Any]:
        """
        Validate chunk sizes are within acceptable range.
        
        Args:
            documents: List of chunked documents
            acceptable_range: Tuple of (min, max) acceptable sizes in characters
            
        Returns:
            Dictionary with validation results
        """
        min_acceptable, max_acceptable = acceptable_range
        
        out_of_range = [
            doc for doc in documents
            if len(doc.page_content) < min_acceptable or
               len(doc.page_content) > max_acceptable
        ]
        
        return {
            "total_chunks": len(documents),
            "out_of_range_chunks": len(out_of_range),
            "percentage_out_of_range": (len(out_of_range) / len(documents) * 100) 
                                       if documents else 0,
            "out_of_range_ids": [doc.metadata.get("chunk_id") for doc in out_of_range],
            "valid": len(out_of_range) == 0,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Document Chunking Module")
    print("=" * 60)
    print("Configuration:")
    print("  - Chunk Size: 600 characters (~150 tokens)")
    print("  - Chunk Overlap: 100 characters (~25 tokens)")
    print("  - Semantic boundaries: Splits on paragraphs, lines, sentences, words")
    print("\nUsage:")
    print("  chunker = DocumentChunker()")
    print("  chunks = chunker.chunk_documents(documents)")
    print("  stats = chunker.get_chunk_stats(chunks)")
    print("  validation = chunker.validate_chunk_sizes(chunks)")
