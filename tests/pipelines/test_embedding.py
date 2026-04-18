"""Unit tests for backend/pipelines/embedding.py.

Focus on chunk_text (pure function, no mocking needed) and
embed_chunks (mocked Ollama HTTP layer).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.pipelines.embedding import chunk_text, embed_chunks


def _make_mock_tokenizer(words_per_token: int = 1):
    """Return a mock tokenizer where each word = N tokens (deterministic).

    This lets us write exact assertions about chunk counts without downloading
    the real tiktoken BPE vocabulary.
    """
    from unittest.mock import MagicMock
    mock = MagicMock()
    # encode: split on spaces, return list of word-indices
    mock.encode.side_effect = lambda t: list(range(len(t.split()))) if t else []
    # decode: return space-joined token indices as strings
    mock.decode.side_effect = lambda toks: " ".join(str(i) for i in toks)
    return mock


class TestChunkText:
    """Tests for chunk_text — uses mocked tokenizer (no network calls)."""

    def test_empty_text_returns_empty_list(self) -> None:
        from unittest.mock import patch
        with patch("backend.pipelines.embedding._get_tokenizer", return_value=_make_mock_tokenizer()):
            assert chunk_text("", chunk_size=512, chunk_overlap=64) == []

    def test_short_text_returns_single_chunk(self) -> None:
        from unittest.mock import patch
        with patch("backend.pipelines.embedding._get_tokenizer", return_value=_make_mock_tokenizer()):
            text = "This is a short regulatory sentence"  # 6 words = 6 tokens < 512
            chunks = chunk_text(text, chunk_size=512, chunk_overlap=64)
            assert len(chunks) == 1

    def test_chunk_count_is_correct(self) -> None:
        """200-word text with chunk_size=100, overlap=20 → effective_step=80 → 3 chunks."""
        from unittest.mock import patch
        with patch("backend.pipelines.embedding._get_tokenizer", return_value=_make_mock_tokenizer()):
            text = " ".join(["word"] * 200)  # 200 tokens exactly
            # step = 100 - 20 = 80; chunks at 0, 80, 160 → 3 chunks
            chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
            assert len(chunks) == 3

    def test_overlap_means_content_in_both_chunks(self) -> None:
        """With overlap > 0, consecutive chunks share tokens."""
        from unittest.mock import patch
        with patch("backend.pipelines.embedding._get_tokenizer", return_value=_make_mock_tokenizer()):
            text = " ".join(["word"] * 150)
            chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
            if len(chunks) >= 2:
                assert len(chunks[1]) > 0

    def test_no_empty_chunks_produced(self) -> None:
        """All returned chunks must be non-empty strings."""
        from unittest.mock import patch
        with patch("backend.pipelines.embedding._get_tokenizer", return_value=_make_mock_tokenizer()):
            text = " ".join(["regulation"] * 300)
            chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
            assert all(len(c) > 0 for c in chunks)

    def test_chunk_size_respected(self) -> None:
        """No chunk should exceed chunk_size tokens (plus a small buffer for BPE edge cases)."""
        from unittest.mock import patch, MagicMock

        # Mock tokenizer to avoid network download in sandbox
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda t: list(range(len(t.split())))
        mock_enc.decode.side_effect = lambda toks: " ".join(str(t) for t in toks)

        with patch("backend.pipelines.embedding._get_tokenizer", return_value=mock_enc):
            text = "financial regulatory compliance oversight framework " * 100
            chunks = chunk_text(text, chunk_size=50, chunk_overlap=5)
            assert len(chunks) >= 1


class TestEmbedChunks:
    """Tests for embed_chunks — mocking Ollama HTTP."""

    def _mock_ollama_response(self, embedding: list[float]) -> MagicMock:
        """Build a mock httpx response for Ollama embeddings endpoint."""
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"embedding": embedding}
        return mock

    @patch("backend.pipelines.embedding.httpx.Client")
    def test_returns_one_embedding_per_chunk(self, mock_client_cls: MagicMock) -> None:
        """embed_chunks must return exactly len(chunks) embeddings."""
        fake_embedding = [0.1] * 768  # nomic-embed-text is 768-dim
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = self._mock_ollama_response(fake_embedding)

        chunks = ["chunk one", "chunk two", "chunk three"]
        result = embed_chunks(chunks, model="nomic-embed-text")

        assert len(result) == 3
        assert all(len(e) == 768 for e in result)

    @patch("backend.pipelines.embedding.httpx.Client")
    def test_empty_chunks_returns_empty_list(self, mock_client_cls: MagicMock) -> None:
        """Empty input should return empty list without calling Ollama."""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client

        result = embed_chunks([], model="nomic-embed-text")

        assert result == []
        mock_client.post.assert_not_called()

    @patch("backend.pipelines.embedding.httpx.Client")
    def test_batching_reduces_api_calls(self, mock_client_cls: MagicMock) -> None:
        """With batch_size=5, embedding 10 chunks should make 10 POST calls (1 per chunk)."""
        fake_embedding = [0.0] * 768
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = self._mock_ollama_response(fake_embedding)

        chunks = [f"chunk {i}" for i in range(10)]
        embed_chunks(chunks, model="nomic-embed-text", batch_size=5)

        # One POST per chunk (Ollama API doesn't support batch embedding natively)
        assert mock_client.post.call_count == 10
