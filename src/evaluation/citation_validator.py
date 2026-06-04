"""
Citation enforcement and hallucination prevention.
Validates answers are grounded in retrieved documents.
"""

import logging
import re
from typing import List, Dict, Tuple
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CitationValidator:
    """Validates citations and prevents hallucinations in answers."""

    def __init__(self, enforce_strict: bool = True):
        """
        Initialize CitationValidator.
        
        Args:
            enforce_strict: If True, reject answers with insufficient citations
        """
        self.enforce_strict = enforce_strict
        logger.info(f"Initialized CitationValidator (strict={enforce_strict})")

    def extract_citations(self, text: str) -> List[int]:
        """
        Extract citation indices from text.
        
        Args:
            text: Text containing citations like [1], [2], etc.
            
        Returns:
            List of citation indices found
        """
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, text)
        return [int(m) for m in matches]

    def validate_answer_grounding(
        self,
        answer: str,
        retrieved_docs: List[Document],
        min_citations: int = 1
    ) -> Dict[str, any]:
        """
        Validate if answer is properly grounded in documents.
        
        Args:
            answer: Generated answer text
            retrieved_docs: Documents used for generation
            min_citations: Minimum required citations
            
        Returns:
            Validation result dictionary
        """
        # Extract citations
        citations = self.extract_citations(answer)
        
        # Check for refusal signals (hallucination prevention)
        refusal_phrases = [
            "i don't have enough information",
            "i could not find",
            "not mentioned in the documents",
            "not available",
            "i don't know",
            "unclear",
        ]
        
        is_refusal = any(
            phrase in answer.lower()
            for phrase in refusal_phrases
        )
        
        # Validate citation indices
        valid_indices = list(range(1, len(retrieved_docs) + 1))
        invalid_citations = [c for c in citations if c not in valid_indices]
        
        # Calculate metrics
        has_sufficient_citations = len(citations) >= min_citations
        all_citations_valid = len(invalid_citations) == 0
        
        return {
            "is_valid": all_citations_valid and has_sufficient_citations and not is_refusal,
            "is_refusal": is_refusal,
            "citations_found": len(citations),
            "citations": citations,
            "invalid_citations": invalid_citations,
            "has_sufficient_citations": has_sufficient_citations,
            "all_citations_valid": all_citations_valid,
            "min_citations_required": min_citations,
        }

    def enforce_citations(
        self,
        answer: str,
        retrieved_docs: List[Document],
        fallback_response: str = None
    ) -> str:
        """
        Enforce citation requirements or return fallback.
        
        Args:
            answer: Generated answer
            retrieved_docs: Retrieved documents
            fallback_response: Response to use if validation fails
            
        Returns:
            Validated answer or fallback
        """
        validation = self.validate_answer_grounding(answer, retrieved_docs)
        
        if not validation["is_valid"] and self.enforce_strict:
            fallback = fallback_response or (
                "I cannot provide a confident answer based on the available documents. "
                "The information needed to answer your question was not found or is unclear."
            )
            logger.warning(f"Citation validation failed, using fallback. Details: {validation}")
            return fallback
        
        return answer

    def get_evidence_for_claims(
        self,
        answer: str,
        retrieved_docs: List[Document]
    ) -> Dict[int, str]:
        """
        Extract evidence for each cited claim.
        
        Args:
            answer: Answer text with citations
            retrieved_docs: Retrieved documents
            
        Returns:
            Dictionary mapping citation index to evidence text
        """
        citations = self.extract_citations(answer)
        evidence = {}
        
        for citation_idx in set(citations):
            if 1 <= citation_idx <= len(retrieved_docs):
                doc = retrieved_docs[citation_idx - 1]
                # Get first 200 characters of evidence
                evidence_text = doc.page_content[:200]
                source = doc.metadata.get("source_file", "unknown")
                evidence[citation_idx] = f"[{source}] {evidence_text}..."
        
        return evidence

    def check_factual_consistency(
        self,
        answer: str,
        retrieved_docs: List[Document]
    ) -> Dict[str, any]:
        """
        Check answer for factual consistency with documents.
        
        Args:
            answer: Answer text
            retrieved_docs: Retrieved documents
            
        Returns:
            Consistency check results
        """
        citations = self.extract_citations(answer)
        
        # Extract sentences/claims from answer
        claim_pattern = r'[^.!?]*[.!?]'
        claims = re.findall(claim_pattern, answer)
        claims = [c.strip() for c in claims if c.strip()]
        
        # For each claim, check if any citation supports it
        uncited_claims = []
        cited_claims = []
        
        for claim in claims:
            # Check if claim contains citation
            if re.search(r'\[\d+\]', claim):
                cited_claims.append(claim)
            else:
                uncited_claims.append(claim)
        
        return {
            "total_claims": len(claims),
            "cited_claims": len(cited_claims),
            "uncited_claims": len(uncited_claims),
            "uncited_claim_list": uncited_claims,
            "citation_rate": len(cited_claims) / len(claims) if claims else 0,
            "needs_improvement": len(uncited_claims) > 0,
        }


class HallucinationDetector:
    """Detects potential hallucinations in generated answers."""

    def __init__(self):
        """Initialize HallucinationDetector."""
        self.hallucination_patterns = [
            r"it is believed that",
            r"supposedly",
            r"allegedly",
            r"it seems like",
            r"it appears to be",
            r"likely",
            r"probably",
            r"might be",
            r"could be",
        ]
        logger.info("Initialized HallucinationDetector")

    def detect(
        self,
        answer: str,
        retrieved_docs: List[Document]
    ) -> Dict[str, any]:
        """
        Detect potential hallucinations.
        
        Args:
            answer: Generated answer
            retrieved_docs: Retrieved documents
            
        Returns:
            Detection results
        """
        # Look for hedging language
        hedging_found = []
        answer_lower = answer.lower()
        
        for pattern in self.hallucination_patterns:
            if re.search(pattern, answer_lower):
                hedging_found.append(pattern)
        
        # Check for specific facts not in documents
        # This is a simplified heuristic check
        doc_content = " ".join([doc.page_content.lower() for doc in retrieved_docs])
        
        # Count citations
        citations = re.findall(r'\[(\d+)\]', answer)
        
        risk_level = "low"
        if len(hedging_found) > 2:
            risk_level = "medium"
        if len(citations) == 0 and len(answer) > 100:
            risk_level = "high"
        
        return {
            "risk_level": risk_level,
            "hedging_language_found": len(hedging_found),
            "hedging_patterns": hedging_found,
            "citations_count": len(citations),
            "likely_hallucination": risk_level == "high",
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Citation Enforcement Module")
    print("=" * 60)
    print("Features:")
    print("  - Citation validation")
    print("  - Hallucination detection")
    print("  - Factual consistency checking")
    print("  - Evidence extraction")
