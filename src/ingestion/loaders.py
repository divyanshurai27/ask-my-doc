"""
Document loaders for various file formats.
Supports: PDF, Markdown, Text, and Web pages.
"""

from pathlib import Path
from typing import List, Optional
import logging

from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
import markdown
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class DocumentLoaders:
    """Load documents from various formats."""

    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """
        Load PDF document using PyPDFLoader.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects (one per page)
        """
        logger.info(f"Loading PDF: {file_path}")
        
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Add metadata
            for i, doc in enumerate(docs, 1):
                doc.metadata["source_type"] = "pdf"
                doc.metadata["source_file"] = Path(file_path).name
                doc.metadata["page_number"] = i
            
            logger.info(f"Successfully loaded {len(docs)} pages from PDF")
            return docs
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise

    @staticmethod
    def load_markdown(file_path: str) -> List[Document]:
        """
        Load Markdown file and parse its content.
        
        Args:
            file_path: Path to Markdown file
            
        Returns:
            List with single Document object
        """
        logger.info(f"Loading Markdown: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Convert markdown to HTML then extract text
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text()
            
            doc = Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "source_type": "markdown",
                    "source_file": Path(file_path).name
                }
            )
            
            logger.info(f"Successfully loaded Markdown file")
            return [doc]
        except Exception as e:
            logger.error(f"Error loading Markdown {file_path}: {str(e)}")
            raise

    @staticmethod
    def load_text(file_path: str) -> List[Document]:
        """
        Load plain text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            List with single Document object
        """
        logger.info(f"Loading text file: {file_path}")
        
        try:
            loader = TextLoader(file_path)
            docs = loader.load()
            
            for doc in docs:
                doc.metadata["source_type"] = "text"
                doc.metadata["source_file"] = Path(file_path).name
            
            logger.info(f"Successfully loaded text file")
            return docs
        except Exception as e:
            logger.error(f"Error loading text {file_path}: {str(e)}")
            raise

    @staticmethod
    def load_web(url: str) -> List[Document]:
        """
        Load web page content.
        
        Args:
            url: URL of web page
            
        Returns:
            List with single Document object
        """
        logger.info(f"Loading web page: {url}")
        
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            
            for doc in docs:
                doc.metadata["source_type"] = "web"
                doc.metadata["source_file"] = url
            
            logger.info(f"Successfully loaded web page")
            return docs
        except Exception as e:
            logger.error(f"Error loading web page {url}: {str(e)}")
            raise

    @staticmethod
    def load_documents(
        file_paths: List[str],
        verbose: bool = True
    ) -> List[Document]:
        """
        Load documents from multiple files.
        Automatically detects file type based on extension.
        
        Args:
            file_paths: List of file paths
            verbose: Print progress messages
            
        Returns:
            List of all loaded documents
        """
        all_docs = []
        
        for file_path in file_paths:
            if verbose:
                logger.info(f"Processing: {file_path}")
            
            path = Path(file_path)
            
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            try:
                # Determine file type and load accordingly
                if path.suffix.lower() == '.pdf':
                    docs = DocumentLoaders.load_pdf(file_path)
                elif path.suffix.lower() in ['.md', '.markdown']:
                    docs = DocumentLoaders.load_markdown(file_path)
                elif path.suffix.lower() in ['.txt']:
                    docs = DocumentLoaders.load_text(file_path)
                else:
                    # Try to load as text by default
                    logger.warning(f"Unknown format {path.suffix}, trying as text")
                    docs = DocumentLoaders.load_text(file_path)
                
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
                continue
        
        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    print("Document Loaders Module")
    print("=" * 50)
    print("Use this module to load documents from:")
    print("  - PDF files")
    print("  - Markdown files")
    print("  - Text files")
    print("  - Web pages")
    print("\nExample:")
    print("  docs = DocumentLoaders.load_documents(['file.pdf', 'file.md'])")
