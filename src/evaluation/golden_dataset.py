"""
Golden dataset for evaluation.
Curated question-answer pairs with document references for quality testing.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class GoldenDataset:
    """Manages golden QA pairs for evaluation."""

    # Sample golden dataset - can be extended
    SAMPLE_QA_PAIRS = [
        {
            "id": "qa_001",
            "question": "What are the company's working hours?",
            "reference_answer": "Employees should work 8 hours per day with breaks totaling 2 hours (30 minutes morning, 30 minutes afternoon, and 1 hour lunch).",
            "expected_sources": ["company_policy.txt"],
            "difficulty": "easy",
            "category": "company_policy",
        },
        {
            "id": "qa_002",
            "question": "How many days of vacation are employees entitled to?",
            "reference_answer": "Employees are entitled to 20 days of paid vacation per year.",
            "expected_sources": ["company_policy.txt"],
            "difficulty": "easy",
            "category": "benefits",
        },
        {
            "id": "qa_003",
            "question": "What benefits are included in the employee benefits package?",
            "reference_answer": "The benefits package includes health insurance, dental coverage, and vision insurance. All employees are automatically enrolled in the company pension plan.",
            "expected_sources": ["benefits.txt"],
            "difficulty": "medium",
            "category": "benefits",
        },
        {
            "id": "qa_004",
            "question": "Does the company offer a matching program?",
            "reference_answer": "Yes, the company provides a 401(k) matching program.",
            "expected_sources": ["benefits.txt"],
            "difficulty": "medium",
            "category": "benefits",
        },
        {
            "id": "qa_005",
            "question": "What information is not available in these documents?",
            "reference_answer": "I could not find enough information in the provided documents to answer this question.",
            "expected_sources": [],
            "difficulty": "hard",
            "category": "refusal",
            "should_refuse": True,
        },
    ]

    def __init__(self, dataset_path: str = None):
        """
        Initialize GoldenDataset.
        
        Args:
            dataset_path: Path to load dataset from
        """
        self.dataset_path = dataset_path
        self.qa_pairs = []
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "total_pairs": 0,
        }
        
        if dataset_path:
            self.load_from_file(dataset_path)
        else:
            self.qa_pairs = [pair.copy() for pair in self.SAMPLE_QA_PAIRS]
            self.metadata["total_pairs"] = len(self.qa_pairs)

    def load_from_file(self, path: str) -> None:
        """
        Load dataset from JSON file.
        
        Args:
            path: Path to JSON file
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.qa_pairs = data.get("qa_pairs", [])
        self.metadata = data.get("metadata", self.metadata)

    def save_to_file(self, path: str) -> None:
        """
        Save dataset to JSON file.
        
        Args:
            path: Path to save to
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "metadata": self.metadata,
            "qa_pairs": self.qa_pairs,
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def add_qa_pair(
        self,
        question: str,
        reference_answer: str,
        expected_sources: List[str] = None,
        difficulty: str = "medium",
        category: str = "general",
        should_refuse: bool = False,
    ) -> str:
        """
        Add a QA pair to the dataset.
        
        Args:
            question: Question text
            reference_answer: Reference answer
            expected_sources: List of relevant source files
            difficulty: Difficulty level (easy, medium, hard)
            category: Question category
            should_refuse: True if answer should be refusal
            
        Returns:
            ID of added pair
        """
        qa_id = f"qa_{len(self.qa_pairs) + 1:03d}"
        
        pair = {
            "id": qa_id,
            "question": question,
            "reference_answer": reference_answer,
            "expected_sources": expected_sources or [],
            "difficulty": difficulty,
            "category": category,
            "should_refuse": should_refuse,
        }
        
        self.qa_pairs.append(pair)
        self.metadata["total_pairs"] = len(self.qa_pairs)
        
        return qa_id

    def get_qa_pairs(self, category: str = None, difficulty: str = None) -> List[Dict]:
        """
        Get QA pairs filtered by category/difficulty.
        
        Args:
            category: Filter by category
            difficulty: Filter by difficulty
            
        Returns:
            Filtered list of QA pairs
        """
        pairs = self.qa_pairs
        
        if category:
            pairs = [p for p in pairs if p.get("category") == category]
        
        if difficulty:
            pairs = [p for p in pairs if p.get("difficulty") == difficulty]
        
        return pairs

    def get_questions(self) -> List[str]:
        """Get all questions."""
        return [pair["question"] for pair in self.qa_pairs]

    def get_reference_answers(self) -> List[str]:
        """Get all reference answers."""
        return [pair["reference_answer"] for pair in self.qa_pairs]

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        categories = {}
        difficulties = {}
        
        for pair in self.qa_pairs:
            cat = pair.get("category", "unknown")
            diff = pair.get("difficulty", "unknown")
            
            categories[cat] = categories.get(cat, 0) + 1
            difficulties[diff] = difficulties.get(diff, 0) + 1
        
        return {
            "total_pairs": len(self.qa_pairs),
            "by_category": categories,
            "by_difficulty": difficulties,
            "refusal_questions": len([p for p in self.qa_pairs if p.get("should_refuse")]),
        }

    def validate(self) -> Dict[str, Any]:
        """
        Validate dataset integrity.
        
        Returns:
            Validation results
        """
        issues = []
        
        for pair in self.qa_pairs:
            if not pair.get("question"):
                issues.append(f"Pair {pair.get('id')} missing question")
            if not pair.get("reference_answer"):
                issues.append(f"Pair {pair.get('id')} missing reference answer")
            if not pair.get("id"):
                issues.append("Pair missing ID")
        
        return {
            "valid": len(issues) == 0,
            "total_pairs": len(self.qa_pairs),
            "issues": issues,
        }


if __name__ == "__main__":
    print("Golden Dataset Module")
    print("=" * 60)
    
    # Create sample dataset
    dataset = GoldenDataset()
    print(f"\nLoaded {len(dataset.qa_pairs)} QA pairs")
    print("\nDataset statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Validate
    validation = dataset.validate()
    print(f"\nValidation: {'✓ PASSED' if validation['valid'] else '✗ FAILED'}")
    if validation['issues']:
        for issue in validation['issues']:
            print(f"  - {issue}")
