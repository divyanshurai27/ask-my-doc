"""
Unit tests for Phase 2 and Phase 3 components.
"""

import os
import pytest
from pathlib import Path
from src.config import Settings, validate_settings
from src.rag.answer_generator import RAGChain, MockLLM
from scripts.check_gates import check_quality_gates


def test_groq_settings_loading():
    """Test that Groq settings are loaded correctly."""
    settings = Settings(
        llm_provider="groq",
        groq_api_key="gsk_test_key_123",
        llm_model="llama-3.1-8b-instant"
    )
    assert settings.llm_provider == "groq"
    assert settings.groq_api_key == "gsk_test_key_123"
    assert settings.llm_model == "llama-3.1-8b-instant"


def test_settings_validation_groq():
    """Test validation settings logic for Groq."""
    settings = Settings(llm_provider="groq", groq_api_key="")
    # Should raise error since groq_api_key is empty and provider is groq
    with pytest.raises(ValueError, match="GROQ_API_KEY is not configured"):
        # We temporarily patch settings inside the settings module
        import sys
        settings_mod = sys.modules["src.config.settings"]
        original_settings = settings_mod.settings
        try:
            settings_mod.settings = settings
            validate_settings()
        finally:
            settings_mod.settings = original_settings


def test_prompt_yaml_exists():
    """Test that config/prompts.yaml exists and is valid."""
    project_root = Path(__file__).parent.parent
    prompts_path = project_root / "config" / "prompts.yaml"
    assert prompts_path.exists()
    
    import yaml
    with open(prompts_path, "r") as f:
        data = yaml.safe_load(f)
    assert "answer_generation" in data
    assert "system_prompt" in data["answer_generation"]
    assert "answer_template" in data["answer_generation"]


def test_rag_chain_with_mock_llm(monkeypatch):
    """Test RAGChain with Mock LLM fallback."""
    from src.config import settings
    monkeypatch.setattr(settings, "groq_api_key", "your-api-key-here")
    
    rag = RAGChain(model_name="llama-3.1-8b-instant")
    assert isinstance(rag.llm, MockLLM)
    
    # Test generation using mock answers
    from langchain_core.documents import Document
    docs = [Document(page_content="Mock doc content", metadata={"source_file": "company_policy.txt", "chunk_id": "chunk_1"})]
    res = rag.generate_answer("What are the working hours?", docs)
    assert "8 hours per day" in res["answer"]
    assert len(res["sources"]) > 0
