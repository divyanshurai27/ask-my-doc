import streamlit as st
import os
import sys
import shutil
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings, Settings
from scripts.demo import RAGPipeline

# Setup logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Ask My Docs - Interactive RAG Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Custom Styling (Vanilla CSS) ---
st.markdown("""
<style>
    /* Custom Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
    
    /* Main Layout Styling */
    .main-header {
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #6B7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card Glassmorphism Styling */
    .metric-card {
        background: rgba(79, 70, 229, 0.08);
        border: 1px solid rgba(79, 70, 229, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Answer Display Container */
    .answer-box {
        background: rgba(79, 70, 229, 0.1);
        border-left: 5px solid #4F46E5;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-success { background-color: #DEF7EC; color: #03543F; }
    .badge-warning { background-color: #FEF3C7; color: #92400E; }
    .badge-info { background-color: #E1EFFE; color: #1E429F; }
    .badge-error { background-color: #FDE8E8; color: #9B1C1C; }
    
</style>
""", unsafe_allow_html=True)

# --- App Directories Setup ---
UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploaded_docs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample_docs"

# Initialize RAG pipeline using streamlit session state caching
@st.cache_resource
def get_pipeline():
    config = Settings()
    pipeline = RAGPipeline(config)
    
    # Pre-populate with sample documents if directory exists
    if SAMPLE_DIR.exists():
        logger.info("Initializing vector store with sample documents...")
        pipeline.ingest_documents(str(SAMPLE_DIR))
        
    return pipeline

try:
    pipeline = get_pipeline()
    pipeline_error = None
except Exception as e:
    pipeline = None
    pipeline_error = e
    logger.exception("Failed to initialize RAG Pipeline")

# --- Sidebar UI ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/bot.png", width=70)
    st.markdown("### **Ask My Docs Configuration**")
    st.caption("Customize the production RAG pipeline settings in real-time.")
    
    st.divider()
    
    # 1. Retrieval Parameters
    st.markdown("#### **1. Retrieval Engine**")
    retrieval_method = st.selectbox(
        "Search Algorithm",
        options=["hybrid", "vector", "bm25"],
        format_func=lambda x: {
            "hybrid": "Hybrid (Sparse + Dense Fusion)",
            "vector": "Dense Vector Search",
            "bm25": "Sparse BM25 Search"
        }[x],
        help="Hybrid search merges keyword and conceptual similarity for maximum recall."
    )
    
    top_k = st.slider("Top K Retrieved Chunks", min_value=1, max_value=10, value=settings.top_k_retrieval)
    
    st.divider()
    
    # 2. Document Uploads
    st.markdown("#### **2. Add Source Documents**")
    st.caption("Upload custom PDF, TXT, or Markdown documents to index them into ChromaDB.")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("🚀 Index Uploaded Files", use_container_width=True):
            with st.spinner("Ingesting and indexing documents..."):
                try:
                    # Save files to disk
                    saved_count = 0
                    for uploaded_file in uploaded_files:
                        # Extract only the base filename to prevent Errno 22 on Windows if full paths are passed
                        safe_filename = Path(uploaded_file.name).name
                        file_path = UPLOAD_DIR / safe_filename
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        saved_count += 1
                    
                    # Run ingestion
                    if saved_count > 0:
                        total_ingested = pipeline.ingest_documents(str(UPLOAD_DIR))
                        st.success(f"Successfully indexed {saved_count} documents into ChromaDB!")
                        st.balloons()
                        # Force refresh stats
                        st.rerun()
                except Exception as ex:
                    logger.exception("Error during ingestion")
                    st.error(f"Error during ingestion: {str(ex)}")
                    st.exception(ex)
                    
    st.divider()
    
    # 3. System Status
    st.markdown("#### **3. Pipeline Status**")
    if pipeline:
        stats = pipeline.get_stats()
        st.success("🟢 Active & Ready")
        st.metric(label="Total Chunks in ChromaDB", value=stats["vector_store"]["total_documents"])
        
        # Display current provider config
        provider_name = settings.llm_provider
        is_mock = "langchain-groq" not in sys.modules and (not settings.groq_api_key or "your-api-key" in settings.groq_api_key)
        
        if is_mock:
            st.warning(f"⚠️ Offline Fallback (Mock LLM)")
        else:
            st.info(f"⚡ LLM Provider: {provider_name.capitalize()}")
    else:
        st.error("🔴 Initialization Failed")

# --- Main Page UI ---
st.markdown('<div class="main-header">🤖 Ask My Docs</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An enterprise-ready, auditable RAG system that answers questions with precise inline citations and automated verification.</div>', unsafe_allow_html=True)

if pipeline_error:
    st.error("### Failed to initialize RAG Pipeline")
    st.exception(pipeline_error)
    st.stop()

# Quick Info Banner on how the pipeline is set up
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card">🎯 <b>Hybrid Search</b><br><small>Keyword + Vector Fusion</small></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card">🔍 <b>Cross-Encoder</b><br><small>SBERT Reranker Model</small></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card">📌 <b>Inline Citations</b><br><small>Strict source grounding</small></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card">🚀 <b>Zero-Hallucination</b><br><small>Confidence Gated Refusals</small></div>', unsafe_allow_html=True)

st.write("")
st.write("")

# --- Ask Question Section ---
st.markdown("### 💬 Ask a Question")
user_query = st.text_input(
    "Enter a query based on the loaded company policy, benefits guide, or your uploaded files:",
    placeholder="e.g., How many days of paid vacation do I get? or Summary of benefits..."
)

# Preset Quick Queries
st.markdown("<small>💡 **Try these sample queries:**</small>", unsafe_allow_html=True)
preset_cols = st.columns(3)
with preset_cols[0]:
    if st.button("📅 What are the company's working hours?", use_container_width=True):
        user_query = "What are the company's working hours?"
with preset_cols[1]:
    if st.button("🌴 How many days of vacation do I get?", use_container_width=True):
        user_query = "How many days of vacation are employees entitled to?"
with preset_cols[2]:
    if st.button("💰 Does the company offer a 401(k) match?", use_container_width=True):
        user_query = "Does the company offer a matching program?"

if user_query:
    with st.spinner("Retrieving document chunks and generating answer..."):
        try:
            # Query the RAG Pipeline
            result = pipeline.query(
                user_query,
                top_k=top_k,
                retrieval_method=retrieval_method
            )
            
            # --- Render RAG Answer ---
            st.markdown("#### **Response Answer**")
            st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
            
            # --- Render Citations & Validation Badge Bar ---
            st.markdown("#### **Verification & Quality Badges**")
            val = result["validation"]
            
            badges_html = ""
            if val["is_valid"]:
                badges_html += '<span class="badge badge-success">✓ Valid Citation Schema</span>'
            else:
                badges_html += '<span class="badge badge-error">✗ Invalid Citation Schema</span>'
                
            if val["grounded"]:
                badges_html += '<span class="badge badge-success">✓ Fully Grounded (No Hallucinations)</span>'
            else:
                badges_html += '<span class="badge badge-warning">⚠️ Grounding Mismatch Found</span>'
                
            if val["is_refusal"]:
                badges_html += '<span class="badge badge-info">ℹ️ System Refusal triggered</span>'
            else:
                badges_html += f'<span class="badge badge-info">ℹ️ Citations Found: {val["citation_count"]}</span>'
                
            badges_html += f'<span class="badge badge-info">ℹ️ Engine: {result["retrieval_method"].upper()}</span>'
            st.markdown(badges_html, unsafe_allow_html=True)
            
            st.write("")
            st.write("")
            
            # --- Display Retrieved Source Materials ---
            st.markdown("#### 📚 Reference Sources")
            for idx, source in enumerate(result["sources"], 1):
                with st.expander(f"Source [{idx}] - {source['source']}"):
                    st.write(f"**Document**: `{source['source']}`")
                    st.write(f"**Chunk Context**:")
                    # Retrieve the content from pipeline retrieved list if not explicitly in result['sources']
                    # Let's search retrieved docs in pipeline
                    content = source.get("content", "")
                    if not content and "retrieved_docs" in locals():
                        pass
                    st.info(source.get("content", "Content text snippet not provided."))
                    
        except Exception as query_err:
            st.error(f"Error querying pipeline: {str(query_err)}")
            st.info("Tip: If you're testing offline, ensure your settings are configured for Mock LLM fallback.")
