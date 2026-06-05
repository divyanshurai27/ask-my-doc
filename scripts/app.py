import streamlit as st
import os
import sys
import logging
from pathlib import Path

# Disable progress bars and force offline model loading for Streamlit compatibility.
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Settings
from scripts.demo import RAGPipeline

# Setup logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Config ---
st.set_page_config(
    page_title="Ask My Docs",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Minimal Premium Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit elements for cleaner look */
    #MainMenu, footer, header { visibility: hidden; }

    /* App container */
    .block-container {
        max-width: 720px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* Title */
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .app-subtitle {
        color: #8b8fa3;
        font-size: 0.95rem;
        margin-bottom: 2.5rem;
    }

    /* Upload area */
    .stFileUploader > div > div {
        border: 2px dashed #d1d5db;
        border-radius: 12px;
        background: #fafbfc;
        transition: border-color 0.2s;
    }
    .stFileUploader > div > div:hover {
        border-color: #6366f1;
    }

    /* Answer card */
    .answer-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f1ff 100%);
        border: 1px solid #e0e2ff;
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        margin-top: 1rem;
        line-height: 1.7;
        color: #1a1a2e;
        font-size: 0.95rem;
    }

    /* Source chip */
    .source-chip {
        display: inline-block;
        background: #eef0ff;
        color: #4f46e5;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 0.2rem 0.3rem 0.2rem 0;
        border: 1px solid #d9ddff;
    }

    /* Section label */
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #9ca3af;
        margin-bottom: 0.6rem;
    }

    /* Status pill */
    .status-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .status-ok { background: #ecfdf5; color: #065f46; }
    .status-err { background: #fef2f2; color: #991b1b; }

    /* Divider */
    .soft-divider {
        height: 1px;
        background: #e5e7eb;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- App Directories ---
UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploaded_docs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample_docs"

# --- Pipeline Init ---
@st.cache_resource
def get_pipeline():
    config = Settings()
    pipeline = RAGPipeline(config)
    if SAMPLE_DIR.exists():
        logger.info("Initializing with sample documents...")
        pipeline.ingest_documents(str(SAMPLE_DIR))
    return pipeline

try:
    pipeline = get_pipeline()
    pipeline_error = None
except Exception as e:
    pipeline = None
    pipeline_error = e
    logger.exception("Failed to initialize RAG Pipeline")

# --- Header ---
st.markdown('<div class="app-title">📄 Ask My Docs</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Upload a PDF and ask questions — get answers with source citations.</div>', unsafe_allow_html=True)

if pipeline_error:
    st.error(f"Pipeline failed to initialize: {pipeline_error}")
    st.stop()

# --- PDF Upload ---
st.markdown('<div class="section-label">Manage Documents</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Drop a PDF here",
        type=["pdf"],
        label_visibility="collapsed",
        key="pdf_upload"
    )
with col2:
    if st.button("🗑️ Clear Database", use_container_width=True, help="Deletes all stored documents from the database"):
        try:
            if pipeline and pipeline.vector_store:
                pipeline.vector_store.delete_collection()
            if "ingested_files" in st.session_state:
                st.session_state.ingested_files = set()
            for f in UPLOAD_DIR.glob("*"):
                if f.is_file() and f.name != ".gitkeep":
                    f.unlink()
            st.success("Database cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear database: {e}")

if uploaded_file:
    safe_filename = Path(uploaded_file.name).name
    file_path = UPLOAD_DIR / safe_filename

    # Only ingest if not already processed this session
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = set()

    if safe_filename not in st.session_state.ingested_files:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Indexing document..."):
            pipeline.ingest_documents(str(UPLOAD_DIR))
        st.session_state.ingested_files.add(safe_filename)
        st.success(f"✓ **{safe_filename}** indexed successfully")
    else:
        st.markdown(f'<span class="status-pill status-ok">✓ {safe_filename} ready</span>', unsafe_allow_html=True)

# --- Divider ---
st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

# --- Question Box ---
st.markdown('<div class="section-label">Ask a Question</div>', unsafe_allow_html=True)
user_query = st.text_input(
    "Your question",
    placeholder="e.g. What are the key skills mentioned in the resume?",
    label_visibility="collapsed"
)

# --- Answer ---
if user_query:
    with st.spinner("Thinking..."):
        try:
            result = pipeline.query(user_query, top_k=5, retrieval_method="hybrid")

            # Answer
            st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-card">{result["answer"]}</div>', unsafe_allow_html=True)

            # Source chips
            if result.get("sources"):
                st.write("")
                st.markdown('<div class="section-label">Sources</div>', unsafe_allow_html=True)
                sources_html = ""
                seen = set()
                for src in result["sources"]:
                    name = src.get("source", "unknown")
                    if name not in seen:
                        sources_html += f'<span class="source-chip">{name}</span>'
                        seen.add(name)
                st.markdown(sources_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")
