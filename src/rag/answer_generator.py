"""
RAG answer generation with citations.
Generates answers from retrieved documents with proper source attribution.
"""

import logging
from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

logger = logging.getLogger(__name__)

# System prompt for answer generation
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided documents.

IMPORTANT RULES:
1. ONLY answer based on the provided documents
2. If the documents don't contain enough information to answer, say: "I could not find enough information in the provided documents."
3. Use inline citations like [1], [2], etc. pointing to sources
4. Be concise and factual
5. Never make up information not in the documents
6. If multiple documents support the answer, cite all of them
7. Format citations as [1], [2], etc. and list sources at the end"""

ANSWER_TEMPLATE = """{system_prompt}

<documents>
{context}
</documents>

Question: {question}

Instructions:
- Answer only based on the documents above
- Use citations [1], [2], etc. for each document
- If unsure, say you don't have enough information
- Format: Answer, then list Sources

Answer:"""


class RAGChain:
    """Generates answers from retrieved documents with citations."""

    def __init__(self, llm=None, model_name: str = "gpt-4o", temperature: float = 0.7):
        """
        Initialize RAG chain.
        
        Args:
            llm: LangChain LLM instance (defaults to OpenAI GPT-4o)
            model_name: Model name (used if llm is None)
            temperature: Temperature for generation
        """
        if llm is None:
            self.llm = OpenAI(model_name=model_name, temperature=temperature)
        else:
            self.llm = llm
        
        # Create prompt template
        self.prompt = PromptTemplate(
            input_variables=["system_prompt", "context", "question"],
            template=ANSWER_TEMPLATE,
        )
        
        # Create chain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        
        logger.info(f"Initialized RAGChain with model: {model_name}")

    def generate_answer(
        self,
        question: str,
        retrieved_docs: List[Document],
        include_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate answer from retrieved documents.
        
        Args:
            question: User question
            retrieved_docs: List of relevant documents
            include_citations: Whether to include citations
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not retrieved_docs:
            return {
                "answer": "I could not find relevant information in the documents.",
                "sources": [],
                "full_response": "No documents were retrieved.",
                "citations_count": 0,
            }
        
        # Prepare context with numbered documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source_file = doc.metadata.get("source_file", "unknown")
            chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")
            
            context_parts.append(
                f"[{i}] Document: {source_file} (Chunk: {chunk_id})\n"
                f"{doc.page_content}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Generate answer
        try:
            response = self.chain.run(
                system_prompt=SYSTEM_PROMPT,
                context=context,
                question=question,
            )
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "answer": "Error generating answer. Please try again.",
                "sources": [],
                "full_response": str(e),
                "citations_count": 0,
            }
        
        # Extract citations and sources
        answer, sources = self._extract_citations_and_sources(
            response, 
            retrieved_docs
        )
        
        # Count citations in answer
        citations_count = answer.count("[") if include_citations else 0
        
        return {
            "answer": answer,
            "sources": sources,
            "full_response": response,
            "citations_count": citations_count,
        }

    def _extract_citations_and_sources(
        self,
        response: str,
        documents: List[Document]
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Extract citations and sources from response.
        
        Args:
            response: Generated response text
            documents: Original retrieved documents
            
        Returns:
            Tuple of (answer_with_citations, sources_list)
        """
        # Split response into answer and sources
        parts = response.split("Sources:")
        answer = parts[0].strip() if parts else response.strip()
        
        # Build sources list
        sources = []
        for i, doc in enumerate(documents, 1):
            source_file = doc.metadata.get("source_file", "unknown")
            chunk_id = doc.metadata.get("chunk_id", "")
            source_type = doc.metadata.get("source_type", "unknown")
            
            sources.append({
                "index": i,
                "source": source_file,
                "chunk_id": chunk_id,
                "type": source_type,
            })
        
        # Append sources to answer if not already present
        if "Sources:" not in response:
            sources_text = "\n".join(
                [f"[{s['index']}] {s['source']} ({s['chunk_id']})" for s in sources]
            )
            answer = f"{answer}\n\nSources:\n{sources_text}"
        
        return answer, sources

    def validate_answer(
        self,
        answer: str,
        retrieved_docs: List[Document],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Validate if answer is grounded in retrieved documents.
        
        Args:
            answer: Generated answer
            retrieved_docs: Documents used for generation
            threshold: Confidence threshold
            
        Returns:
            Validation results
        """
        # Check for refusal signals
        refusal_phrases = [
            "i don't have enough information",
            "i could not find",
            "not mentioned",
            "not available",
        ]
        
        is_refusal = any(
            phrase in answer.lower() 
            for phrase in refusal_phrases
        )
        
        # Count citations
        citation_count = answer.count("[")
        has_citations = citation_count > 0
        
        # Basic validation
        validation = {
            "is_valid": not is_refusal and has_citations,
            "is_refusal": is_refusal,
            "has_citations": has_citations,
            "citation_count": citation_count,
            "grounded": has_citations,  # Simple heuristic
        }
        
        return validation


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("RAG Answer Generation Module")
    print("=" * 60)
    print("Features:")
    print("  - LangChain-based answer generation")
    print("  - Citation extraction and validation")
    print("  - Source attribution")
    print("  - Hallucination prevention")
