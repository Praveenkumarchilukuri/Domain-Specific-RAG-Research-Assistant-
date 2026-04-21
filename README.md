# 🔬 Domain-specific RAG Research Assistant

> Build a retrieval-augmented generation system that answers questions over a corpus of real academic papers (e.g., ArXiv PDFs on a topic you care about).

**Resume headline**: *"Built RAG pipeline over 500+ papers, reducing hallucinations by 40% vs baseline"*

## 🏗️ Architecture

```
PDF Papers → PyMuPDF Parser → Text Chunking → Embeddings → ChromaDB
                                                              ↓
User Question → Gemini Embedding → Similarity Search → Reranking
                                                              ↓
                                           Top-K Chunks → Gemini 1.5 Pro → Answer + Citations
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up environment
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
# Get one free at https://aistudio.google.com/apikey
```

### 3. Add your papers
Place PDF files in the `data/papers/` directory.

### 4. Run the app
```bash
streamlit run app.py
```

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web UI (Q&A, evaluation) |
| `config.py` | Central configuration from `.env` |
| `src/ingestion.py` | PDF parsing + text chunking |
| `src/embeddings.py` | Gemini / HuggingFace embedding models |
| `src/vector_store.py` | ChromaDB + FAISS vector storage |
| `src/retriever.py` | Similarity search + cross-encoder reranking |
| `src/llm.py` | Gemini LLM with RAG prompt |
| `src/evaluation.py` | Faithfulness / relevance metrics |

## 🔧 Tech Stack

- **Python** + **LangChain** — orchestration framework
- **Google Gemini 1.5 Pro** — LLM for generation
- **Google Gemini Embeddings** — text-to-vector conversion
- **ChromaDB** — persistent vector database
- **FAISS** — alternative vector search (Facebook AI)
- **PyMuPDF** — PDF text extraction
- **Cross-Encoder Reranking** — improves retrieval accuracy
- **Streamlit** — web interface
- **LLM-as-Judge Evaluation** — measures answer quality

## 📊 Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| Faithfulness | Is the answer grounded in the sources? (no hallucinations) |
| Answer Relevance | Does the answer address the question? |
| Context Relevance | Were the retrieved chunks actually relevant? |

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

## 🔑 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | (required) | Your Google AI Studio API key |
| `EMBEDDING_PROVIDER` | `gemini` | `gemini` or `huggingface` |
| `LLM_MODEL` | `gemini-1.5-pro` | LLM model name |
| `CHUNK_SIZE` | `512` | Characters per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `5` | Number of docs to retrieve |
| `USE_RERANKING` | `true` | Enable cross-encoder reranking |

## 📄 License

MIT
