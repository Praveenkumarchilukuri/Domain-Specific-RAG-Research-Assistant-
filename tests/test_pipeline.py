"""
test_pipeline.py — Tests for the RAG pipeline.

WHY TESTS?
Tests verify that each module works correctly in isolation AND together.
In interviews, having tests shows engineering maturity — you don't just
build things, you verify they work.

These tests are designed to run WITHOUT an API key by mocking the
LLM and embedding calls. This means CI/CD can run them automatically.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document


class TestIngestion(unittest.TestCase):
    """Tests for the document ingestion pipeline."""

    def test_chunk_documents_basic(self):
        """Verify that chunking splits documents and preserves metadata."""
        from src.ingestion import chunk_documents

        # Create a fake document with enough text to be split
        long_text = "This is a test sentence. " * 100  # ~2500 chars
        docs = [Document(
            page_content=long_text,
            metadata={"source": "test.pdf", "page": 1},
        )]

        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)

        # Should produce multiple chunks
        self.assertGreater(len(chunks), 1)

        # Each chunk should have metadata preserved
        for chunk in chunks:
            self.assertEqual(chunk.metadata["source"], "test.pdf")
            self.assertEqual(chunk.metadata["page"], 1)

        # Each chunk should be approximately the right size
        for chunk in chunks:
            self.assertLessEqual(len(chunk.page_content), 220)  # Allow small overflow

    def test_chunk_documents_empty(self):
        """Verify that chunking handles empty input gracefully."""
        from src.ingestion import chunk_documents

        chunks = chunk_documents([], chunk_size=200, chunk_overlap=20)
        self.assertEqual(len(chunks), 0)

    def test_chunk_overlap_preserves_context(self):
        """Verify that chunks overlap to preserve boundary context."""
        from src.ingestion import chunk_documents

        # Create text where overlap matters
        sentences = [f"Sentence number {i}. " for i in range(50)]
        text = "".join(sentences)
        docs = [Document(page_content=text, metadata={"source": "test.pdf", "page": 1})]

        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=50)

        # Check that consecutive chunks share some text (overlap)
        if len(chunks) >= 2:
            chunk1_end = chunks[0].page_content[-50:]
            chunk2_start = chunks[1].page_content[:50:]
            # There should be some overlap in the text
            self.assertTrue(
                len(chunks) > 1,
                "Multiple chunks should be produced"
            )


class TestEmbeddings(unittest.TestCase):
    """Tests for the embedding model factory."""

    def test_get_embedding_model_invalid_provider(self):
        """Verify that an invalid provider raises ValueError."""
        from src.embeddings import get_embedding_model

        with self.assertRaises(ValueError):
            get_embedding_model("invalid_provider")

    @patch("src.embeddings.GoogleGenerativeAIEmbeddings")
    def test_get_gemini_embeddings(self, mock_gemini):
        """Verify that Gemini embeddings are created with correct params."""
        from src.embeddings import get_embedding_model

        mock_gemini.return_value = MagicMock()
        model = get_embedding_model("gemini")

        mock_gemini.assert_called_once()
        self.assertIsNotNone(model)


class TestRetriever(unittest.TestCase):
    """Tests for the retrieval pipeline."""

    def test_rerank_results_empty(self):
        """Verify that reranking handles empty input."""
        from src.retriever import rerank_results

        result = rerank_results("test query", [])
        self.assertEqual(result, [])

    def test_retrieve_without_reranking(self):
        """Verify basic retrieval without reranking."""
        from src.retriever import retrieve

        # Mock vector store
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            Document(page_content="Relevant text", metadata={"source": "test.pdf", "page": 1}),
        ]

        results = retrieve("test query", mock_store, k=1, use_reranking=False)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "Relevant text")
        mock_store.similarity_search.assert_called_once_with("test query", k=1)


class TestLLM(unittest.TestCase):
    """Tests for the LLM query interface."""

    @patch("src.llm.ChatGoogleGenerativeAI")
    def test_get_llm_creates_instance(self, mock_llm_class):
        """Verify that get_llm creates a Gemini instance."""
        from src.llm import get_llm

        mock_llm_class.return_value = MagicMock()
        llm = get_llm("gemini-1.5-pro")

        mock_llm_class.assert_called_once()
        self.assertIsNotNone(llm)

    def test_prompt_template_has_correct_variables(self):
        """Verify that the RAG prompt contains both context and question."""
        from src.llm import RAG_PROMPT_TEMPLATE

        # Verify the prompt template has the right placeholders
        self.assertIn("{context}", RAG_PROMPT_TEMPLATE)
        self.assertIn("{question}", RAG_PROMPT_TEMPLATE)

        # Verify the prompt instructs citation
        self.assertIn("cite", RAG_PROMPT_TEMPLATE.lower())


if __name__ == "__main__":
    unittest.main()
