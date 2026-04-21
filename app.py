"""
app.py — Streamlit web UI for the RAG Research Assistant.

WHY STREAMLIT?
Streamlit lets us build a professional-looking web interface with pure
Python — no HTML/CSS/JavaScript needed. It's the go-to framework for
ML/AI demos because:
  1. You can prototype in hours, not weeks
  2. It handles state management, file uploads, real-time updates
  3. Interviewers and recruiters can immediately interact with your project
  4. It deploys easily to Streamlit Cloud for free public demos

THE UI HAS THREE TABS:
  Tab 1 — Q&A: Ask questions, get answers with source citations
  Tab 2 — Model Comparison: Compare different LLM responses side-by-side
  Tab 3 — Evaluation: See faithfulness/relevance scores for each answer
"""

import os
import sys
import streamlit as st
import pandas as pd

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.ingestion import load_all_pdfs, load_pdf, chunk_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_chroma_store, load_chroma_store, get_existing_sources, add_to_chroma_store
from src.retriever import retrieve
from src.llm import get_llm, query_with_context
from src.evaluation import full_evaluation


# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS for a polished look
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    /* Source citation cards */
    .source-card {
        background: #1e1e2e;
        border-left: 4px solid #667eea;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    
    /* Status badges */
    .status-ready { color: #10b981; font-weight: bold; }
    .status-empty { color: #f59e0b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State Initialization
# WHY SESSION STATE?
# Streamlit reruns the entire script on every interaction.
# Session state persists data across reruns so we don't
# re-embed documents every time the user clicks a button.
# ─────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None


# ─────────────────────────────────────────────
# Sidebar — Configuration & Document Ingestion
# WHY A SIDEBAR?
# Separates setup/config controls from the main Q&A area.
# Users configure once (sidebar) then interact repeatedly (main).
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # API Key input
    api_key = st.text_input(
        "Google API Key",
        value=Config.GOOGLE_API_KEY,
        type="password",
        help="Get your key at https://aistudio.google.com/apikey",
    )
    if api_key:
        Config.GOOGLE_API_KEY = api_key
        os.environ["GOOGLE_API_KEY"] = api_key

    st.divider()

    # Embedding provider selection
    embedding_provider = st.selectbox(
        "Embedding Model",
        ["gemini", "huggingface"],
        index=["gemini", "huggingface"].index(Config.EMBEDDING_PROVIDER),
        help="Gemini: fast, cloud-based. HuggingFace: free, runs locally.",
    )

    # Retrieval settings
    top_k = st.slider("Documents to retrieve", 1, 20, Config.TOP_K)
    use_reranking = st.checkbox("Enable reranking", value=Config.USE_RERANKING,
                                 help="Improves accuracy but adds ~1s latency")

    st.divider()

    # ── Document Ingestion ──
    st.markdown("## 📄 Document Ingestion")

    # File uploader for PDFs
    uploaded_files = st.file_uploader(
        "Upload research papers",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload ArXiv PDFs or any academic papers",
    )

    if uploaded_files:
        if st.button("📥 Ingest Uploaded PDFs", use_container_width=True):
            with st.spinner("Processing PDFs..."):
                # Save uploaded files to papers directory
                os.makedirs(Config.PAPERS_DIR, exist_ok=True)
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(Config.PAPERS_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"✓ Saved {uploaded_file.name}")

    st.divider()

    # ── Paper Selector — pick which PDFs to ingest ──
    os.makedirs(Config.PAPERS_DIR, exist_ok=True)
    available_pdfs = sorted(
        [f for f in os.listdir(Config.PAPERS_DIR) if f.lower().endswith(".pdf")]
    )

    if available_pdfs:
        selected_pdfs = st.multiselect(
            "Select papers to ingest",
            available_pdfs,
            default=available_pdfs,
            help="Choose which PDFs from data/papers/ to load into the vector store",
        )

        if st.button("🔄 Ingest Selected Papers", use_container_width=True,
                      disabled=not selected_pdfs):
            with st.spinner(f"Loading {len(selected_pdfs)} paper(s)..."):
                # Check which papers are already in the store
                existing_sources = set()
                if st.session_state.vector_store:
                    existing_sources = get_existing_sources(st.session_state.vector_store)

                new_pdfs = [p for p in selected_pdfs if p not in existing_sources]
                skipped_pdfs = [p for p in selected_pdfs if p in existing_sources]

                # Notify about skipped papers
                for pdf_file in skipped_pdfs:
                    st.info(f"⏭️ {pdf_file} — already in vector store, skipped")

                if not new_pdfs:
                    st.success("All selected papers are already in the vector store!")
                else:
                    # Load only new papers
                    documents = []
                    for pdf_file in new_pdfs:
                        file_path = os.path.join(Config.PAPERS_DIR, pdf_file)
                        try:
                            docs = load_pdf(file_path)
                            documents.extend(docs)
                            st.success(f"✓ Loaded {pdf_file} ({len(docs)} pages)")
                        except Exception as e:
                            st.error(f"✗ Error loading {pdf_file}: {e}")

                    if documents:
                        chunks = chunk_documents(documents)
                        st.session_state.embedding_model = get_embedding_model(embedding_provider)

                        if st.session_state.vector_store:
                            # Append to existing store
                            st.session_state.vector_store = add_to_chroma_store(
                                chunks, st.session_state.vector_store
                            )
                        else:
                            # Create new store
                            st.session_state.vector_store = create_chroma_store(
                                chunks, st.session_state.embedding_model
                            )
                        st.success(f"✓ Added {len(chunks)} chunks from {len(new_pdfs)} new paper(s)!")
                    else:
                        st.warning("No content extracted from selected papers.")
    else:
        st.info("No PDFs found. Upload papers above or add them to `data/papers/`.")

    # Load existing store
    if st.button("📂 Load existing vector store", use_container_width=True):
        try:
            st.session_state.embedding_model = get_embedding_model(embedding_provider)
            st.session_state.vector_store = load_chroma_store(
                st.session_state.embedding_model
            )
            st.success("✓ Loaded existing vector store!")
        except Exception as e:
            st.error(f"No existing store found: {e}")

    # Status indicator
    st.divider()
    if st.session_state.vector_store:
        st.markdown('<p class="status-ready">● Vector store ready</p>',
                    unsafe_allow_html=True)

        # Show what's in the vector store
        try:
            collection = st.session_state.vector_store._collection
            total_chunks = collection.count()
            all_meta = collection.get()["metadatas"]
            sources = sorted(set(m.get("source", "Unknown") for m in all_meta))

            st.caption(f"**{total_chunks} chunks** from **{len(sources)} paper(s)**:")
            for src in sources:
                chunk_count = sum(1 for m in all_meta if m.get("source") == src)
                st.caption(f"  📄 {src} ({chunk_count} chunks)")
        except Exception:
            st.caption("Vector store loaded (unable to read details)")

        # Clear button
        if st.button("🗑️ Clear Vector Store", use_container_width=True, type="secondary"):
            import shutil
            try:
                shutil.rmtree(Config.CHROMA_PERSIST_DIR, ignore_errors=True)
            except Exception:
                pass
            st.session_state.vector_store = None
            st.session_state.embedding_model = None
            st.session_state.chat_history = []
            st.success("✓ Vector store and chat history cleared!")
            st.rerun()
    else:
        st.markdown('<p class="status-empty">○ No vector store loaded</p>',
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main Area — Title
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🔬 RAG Research Assistant</p>',
            unsafe_allow_html=True)
st.markdown("Ask questions over your corpus of academic papers. "
            "Powered by Google Gemini + ChromaDB.")

# ─────────────────────────────────────────────
# Tabs: Q&A | Comparison | Evaluation
# WHY TABS?
# Keeps the UI clean. Most of the time users are in Q&A mode.
# Comparison and Evaluation are advanced features that should
# be accessible but not clutter the main interface.
# ─────────────────────────────────────────────
tab_qa, tab_eval = st.tabs(["💬 Q&A", "📊 Evaluation"])


# ──────── TAB 1: QUESTION & ANSWER ────────
with tab_qa:
    # Chat history display
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show sources for assistant messages
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 View Sources"):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<strong>{src["source"]}</strong> — Page {src["page"]}<br>'
                            f'{src["text"][:200]}...'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # Question input
    question = st.chat_input("Ask a question about your papers...")

    if question:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        if st.session_state.vector_store is None:
            with st.chat_message("assistant"):
                st.warning("⚠️ Please ingest documents first (use the sidebar).")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "⚠️ Please ingest documents first.",
                })
        else:
            with st.chat_message("assistant"):
                with st.spinner("🔍 Retrieving relevant passages & generating answer..."):
                    try:
                        # Step 1: Retrieve relevant chunks
                        docs = retrieve(
                            question,
                            st.session_state.vector_store,
                            k=top_k,
                            use_reranking=use_reranking,
                        )

                        # Step 2: Query LLM with context
                        result = query_with_context(question, docs)

                        # Display answer
                        st.markdown(result["answer"])
                        st.caption(f"_Model: {result['model']}_")

                        # Prepare source info
                        sources = [
                            {
                                "source": doc.metadata.get("source", "Unknown"),
                                "page": doc.metadata.get("page", "N/A"),
                                "text": doc.page_content,
                            }
                            for doc in result["source_documents"]
                        ]

                        # Show sources in expander
                        with st.expander("📚 View Sources"):
                            for src in sources:
                                st.markdown(
                                    f'<div class="source-card">'
                                    f'<strong>{src["source"]}</strong> — Page {src["page"]}<br>'
                                    f'{src["text"][:200]}...'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                        # Save to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": sources,
                        })

                    except Exception as e:
                        error_msg = str(e)
                        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                            st.error(
                                "⏳ **Rate limit exceeded.** Your Google API free-tier quota "
                                "is exhausted. Options:\n"
                                "1. **Wait** — quota resets every minute/day\n"
                                "2. **Switch to HuggingFace** embedding (sidebar) to reduce API calls\n"
                                "3. **Upgrade** your Google AI plan at https://ai.google.dev\n"
                                f"\n*Model: {Config.LLM_MODEL}*"
                            )
                        else:
                            st.error(f"❌ Error: {error_msg}")


# ──────── TAB 2: EVALUATION ────────
with tab_eval:
    st.markdown("### 📊 Answer Quality Evaluation")
    st.markdown(
        "Evaluate the quality of RAG responses using three metrics: "
        "**Faithfulness** (is the answer grounded?), "
        "**Answer Relevance** (does it address the question?), "
        "**Context Relevance** (was retrieval good?)."
    )

    eval_question = st.text_input("Question to evaluate", key="eval_q")
    eval_answer = st.text_area("Answer to evaluate", key="eval_a", height=150)

    if st.button("🔬 Run Evaluation", disabled=not (eval_question and eval_answer)):
        if st.session_state.vector_store is None:
            st.warning("Please ingest documents first.")
        else:
            with st.spinner("Running evaluation (this calls the LLM 3 times)..."):
                try:
                    # Retrieve context for the question
                    eval_docs = retrieve(
                        eval_question,
                        st.session_state.vector_store,
                        k=top_k,
                        use_reranking=use_reranking,
                    )

                    # Run evaluation
                    scores = full_evaluation(eval_question, eval_answer, eval_docs)

                    # Display scores as columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Faithfulness", f"{scores['faithfulness']:.2f}")
                        st.progress(scores["faithfulness"])
                    with col2:
                        st.metric("Answer Relevance", f"{scores['answer_relevance']:.2f}")
                        st.progress(scores["answer_relevance"])
                    with col3:
                        st.metric("Context Relevance", f"{scores['context_relevance']:.2f}")
                        st.progress(scores["context_relevance"])

                    # Interpretation
                    avg_score = sum(scores.values()) / len(scores)
                    if avg_score >= 0.8:
                        st.success(f"✅ Overall quality: Excellent ({avg_score:.2f})")
                    elif avg_score >= 0.5:
                        st.warning(f"⚠️ Overall quality: Moderate ({avg_score:.2f})")
                    else:
                        st.error(f"❌ Overall quality: Poor ({avg_score:.2f})")

                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                        st.error(
                            "⏳ **Rate limit exceeded on all models.** "
                            "Please wait a few minutes and try again."
                        )
                    else:
                        st.error(f"❌ Evaluation error: {error_msg}")


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<center style='color: #666; font-size: 0.8rem;'>"
    "Built with LangChain + Google Gemini + ChromaDB + Streamlit | "
    "Domain-specific RAG Research Assistant"
    "</center>",
    unsafe_allow_html=True,
)
