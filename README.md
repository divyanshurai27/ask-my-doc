# Ask My Docs - Production-Grade RAG System

A domain-specific Retrieval-Augmented Generation (RAG) system that retrieves relevant information from documents and provides answers with proper citations to ensure trustworthiness.

## Features

- **Multi-format Document Support**: Load PDF, Markdown, and web pages
- **Intelligent Chunking**: 500-800 token chunks with ~100 token overlap
- **Hybrid Retrieval**: Combines BM25 keyword search and vector similarity
- **Cross-Encoder Reranking**: Improves precision of retrieved chunks
- **Citation Enforcement**: Ensures answers are grounded in source documents
- **Quality Metrics**: RAGAS evaluation with CI/CD gating
- **No Hallucination**: System refuses to answer when confidence is low

## Project Structure

```
ask-my-docs/
├── src/
│   ├── ingestion/          # Document loading and chunking
│   ├── storage/            # Vector store (ChromaDB)
│   ├── retrieval/          # Retrieval pipelines (BM25, vector, hybrid)
│   ├── rag/                # RAG chain and answer generation
│   ├── evaluation/         # RAGAS evaluation
│   └── config.py           # Configuration management
├── tests/                  # Unit and integration tests
├── scripts/                # Utility scripts (demo, evaluate, etc)
├── data/
│   └── sample_docs/        # Sample documents for testing
├── config/
│   └── prompts.yaml        # Versioned prompt templates
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
└── README.md              # This file
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- OpenAI API key (or alternative LLM)
- 8GB+ RAM

### Installation

1. **Clone and navigate to project**:
```bash
cd ask-my-docs
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. **Verify setup**:
```bash
python src/config.py
```

You should see your configuration printed.

## Implementation Phases

### Phase 1: Fundamentals (Days 1-4)
✅ Document loading and chunking
✅ Vector embeddings (ChromaDB)
✅ Basic retrieval
✅ RAG with citations

### Phase 2: Production Ready (Days 5-9)
- BM25 keyword search
- Hybrid retrieval fusion
- Cross-encoder reranking
- Citation enforcement
- Prompt versioning

### Phase 3: Evaluation & CI (Days 10-12)
- Golden dataset creation
- RAGAS evaluation metrics
- GitHub Actions CI/CD

## Quick Start

### Phase 1 Demo
```bash
python scripts/demo.py
```

### Run Tests
```bash
pytest tests/ -v
```

### Evaluate Golden Dataset
```bash
python scripts/evaluate.py
```

### Check Quality Gates
```bash
python scripts/check_gates.py
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangChain |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers |
| Keyword Search | rank-bm25 |
| Reranking | SBERT Cross-Encoder |
| LLM | OpenAI GPT-4o |
| Evaluation | RAGAS |
| CI/CD | GitHub Actions |

## Success Criteria

- ✅ Faithfulness > 0.85 (no hallucinations)
- ✅ Answer Relevancy > 0.80
- ✅ Context Precision > 0.75
- ✅ Context Recall > 0.70
- ✅ CI/CD blocks PRs that fail quality gates

## Documentation

- `RAG_IMPLEMENTATION_PLAN.md` - Detailed implementation guide
- `QUICK_REFERENCE.md` - Visual guide and checklist
- `PHASE1_CODE_EXAMPLES.md` - Code templates

## Next Steps

1. Read the implementation plan
2. Complete Phase 1-1 to Phase 1-7
3. Move to Phase 2 for production enhancements
4. Implement Phase 3 for evaluation and CI/CD

## Development

### Code Style
- Use Python 3.8+ syntax
- Follow PEP 8 guidelines
- Type hints for all functions

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_loaders.py -v
```

### Debugging
Set `DEBUG=true` in `.env` for verbose logging:
```
DEBUG=true
LOG_LEVEL=DEBUG
```

## Troubleshooting

### ChromaDB Issues
```bash
# Reset ChromaDB
rm -rf data/chroma_db
```

### Missing Dependencies
```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade
```

### API Key Issues
- Verify `OPENAI_API_KEY` is set in `.env`
- Test with: `python -c "from openai import OpenAI; print('OK')"`

## Contributing

Follow the implementation plan in `RAG_IMPLEMENTATION_PLAN.md` for task breakdown.

## License

MIT

## Timeline

- **Days 1-4**: Phase 1 (Fundamentals)
- **Days 5-9**: Phase 2 (Production Ready)
- **Days 10-12**: Phase 3 (Evaluation & CI)

**Total**: 12 days to production-ready system

---

**Status**: ✅ Ready to implement
**Last Updated**: 2026-06-04
