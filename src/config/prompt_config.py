"""
Prompt configuration management.
Versioned, configurable prompts for RAG system.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class PromptConfig:
    """Manages versioned RAG prompts."""

    DEFAULT_PROMPTS = {
        "system_prompt": """You are a helpful assistant that answers questions based on provided documents.

IMPORTANT RULES:
1. ONLY answer based on the provided documents
2. If the documents don't contain enough information to answer, say: "I could not find enough information in the provided documents."
3. Use inline citations like [1], [2], etc. pointing to sources
4. Be concise and factual
5. Never make up information not in the documents
6. If multiple documents support the answer, cite all of them
7. Format citations as [1], [2], etc. and list sources at the end""",

        "answer_prompt": """{system_prompt}

<documents>
{context}
</documents>

Question: {question}

Instructions:
- Answer only based on the documents above
- Use citations [1], [2], etc. for each document
- If unsure, say you don't have enough information
- Format: Answer, then list Sources

Answer:""",

        "refusal_prompt": """I could not find enough information in the provided documents to answer your question: {question}

To help you better, I would need documents containing information about {topic}.""",

        "query_expansion_prompt": """Given the user question: {question}

Generate 3 alternative phrasings or related queries that could help find more relevant documents:

1. 
2. 
3. """,

        "summary_prompt": """Based on the following documents, provide a concise summary:

{context}

Summary:""",

        "context_compression_prompt": """Given the context below, extract only the most relevant information to answer the question: {question}

Context:
{context}

Compressed context (keep only essential information):""",
    }

    def __init__(self, config_path: str = None):
        """
        Initialize PromptConfig.
        
        Args:
            config_path: Path to load custom prompts from
        """
        self.config_path = config_path
        self.prompts = self.DEFAULT_PROMPTS.copy()
        self.version = "1.0"
        self.last_updated = datetime.now().isoformat()
        
        if config_path:
            self.load_from_file(config_path)
        
        logger.info(f"Initialized PromptConfig (version={self.version})")

    def load_from_file(self, config_path: str) -> None:
        """
        Load prompts from JSON file.
        
        Args:
            config_path: Path to JSON config file
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return
        
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            
            if "prompts" in config:
                self.prompts.update(config["prompts"])
            if "version" in config:
                self.version = config["version"]
            
            self.last_updated = datetime.now().isoformat()
            logger.info(f"Loaded prompts from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")

    def save_to_file(self, config_path: str) -> None:
        """
        Save prompts to JSON file.
        
        Args:
            config_path: Path to save JSON config to
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            "version": self.version,
            "last_updated": self.last_updated,
            "prompts": self.prompts,
        }
        
        try:
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved prompts to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")

    def get_prompt(self, name: str, **kwargs) -> str:
        """
        Get a prompt template and format it.
        
        Args:
            name: Prompt name
            **kwargs: Template variables
            
        Returns:
            Formatted prompt string
        """
        if name not in self.prompts:
            logger.warning(f"Prompt '{name}' not found, using default")
            return self.prompts.get("system_prompt", "")
        
        prompt_template = self.prompts[name]
        
        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing template variable: {str(e)}")
            return prompt_template

    def set_prompt(self, name: str, content: str) -> None:
        """
        Set a custom prompt.
        
        Args:
            name: Prompt name
            content: Prompt content
        """
        self.prompts[name] = content
        self.last_updated = datetime.now().isoformat()
        logger.info(f"Updated prompt: {name}")

    def list_prompts(self) -> Dict[str, str]:
        """
        List all available prompts.
        
        Returns:
            Dictionary of prompt names and short descriptions
        """
        descriptions = {
            "system_prompt": "System instructions for the assistant",
            "answer_prompt": "Main prompt for answer generation",
            "refusal_prompt": "Prompt for refusal responses",
            "query_expansion_prompt": "Prompt for expanding queries",
            "summary_prompt": "Prompt for document summarization",
            "context_compression_prompt": "Prompt for context compression",
        }
        
        return {
            name: descriptions.get(name, "Custom prompt")
            for name in self.prompts.keys()
        }

    def validate_prompt(self, name: str) -> Dict[str, Any]:
        """
        Validate a prompt template for required variables.
        
        Args:
            name: Prompt name
            
        Returns:
            Validation results
        """
        if name not in self.prompts:
            return {"valid": False, "error": f"Prompt '{name}' not found"}
        
        template = self.prompts[name]
        
        # Extract variables
        import string
        formatter = string.Formatter()
        variables = [field_name for _, field_name, _, _ in formatter.parse(template) if field_name]
        
        return {
            "valid": True,
            "name": name,
            "variables": list(set(variables)),
            "template_length": len(template),
        }

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get summary of configuration.
        
        Returns:
            Configuration summary
        """
        return {
            "version": self.version,
            "last_updated": self.last_updated,
            "total_prompts": len(self.prompts),
            "prompt_names": list(self.prompts.keys()),
        }

    def reset_to_defaults(self) -> None:
        """Reset all prompts to defaults."""
        self.prompts = self.DEFAULT_PROMPTS.copy()
        self.last_updated = datetime.now().isoformat()
        logger.info("Reset prompts to defaults")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Prompt Configuration Module")
    print("=" * 60)
    print("Features:")
    print("  - Versioned prompts")
    print("  - JSON config file support")
    print("  - Template variable validation")
    print("  - Easy customization")
    
    # Example usage
    config = PromptConfig()
    print("\nAvailable prompts:")
    for name, desc in config.list_prompts().items():
        print(f"  - {name}: {desc}")
