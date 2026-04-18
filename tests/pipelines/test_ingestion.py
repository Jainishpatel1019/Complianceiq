"""Unit tests for backend/pipelines/ingestion.py.

Tests use unittest.mock to avoid real HTTP calls and DB connections.
Why mock at the HTTP layer (not the function layer): we want to test our
parsing and error-handling logic, not just that we called the function.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.pipelines.ingestion import (
    _map_doc_type,
    _parse_date,
    fetch_federal_register,
)


class TestParseDate:
    """Tests for the _parse_date helper."""

    def test_valid_date_returns_datetime(self) -> None:
        result = _parse_date("2024-01-15")
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_none_input_returns_none(self) -> None:
        assert _parse_date(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_date("") is None

    def test_invalid_format_returns_none(self) -> None:
        # Should not raise — returns None and logs a warning
        assert _parse_date("not-a-date") is None

    def test_different_valid_date(self) -> None:
        result = _parse_date("1994-03-31")
        assert result is not None
        assert result.year == 1994


class TestMapDocType:
    """Tests for the _map_doc_type helper."""

    def test_rule_maps_to_final_rule(self) -> None:
        assert _map_doc_type("Rule") == "final_rule"

    def test_proposed_rule_maps_correctly(self) -> None:
        assert _map_doc_type("Proposed Rule") == "proposed_rule"

    def test_notice_maps_correctly(self) -> None:
        assert _map_doc_type("Notice") == "notice"

    def test_unknown_type_returns_none(self) -> None:
        assert _map_doc_type("Unknown Type") is None

    def test_none_returns_none(self) -> None:
        assert _map_doc_type(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _map_doc_type("") is None


class TestFetchFederalRegister:
    """Tests for fetch_federal_register — mocking the HTTP layer."""

    def _make_mock_response(self, results: list[dict], next_url: str | None = None) -> MagicMock:
        """Helper to build a mock httpx response."""
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {
            "results": results,
            "next_page_url": next_url,
        }
        return mock

    @patch("backend.pipelines.ingestion.httpx.Client")
    def test_returns_documents_with_source_field(self, mock_client_cls: MagicMock) -> None:
        """Each returned document must have source='federal_register'."""
        sample_doc = {"document_number": "2024-00001", "title": "Test Rule", "type": "Rule"}
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.get.return_value = self._make_mock_response([sample_doc])

        results = fetch_federal_register(since_date="2024-01-01")

        assert len(results) == 1
        assert results[0]["source"] == "federal_register"
        assert results[0]["document_number"] == "2024-00001"

    @patch("backend.pipelines.ingestion.httpx.Client")
    def test_paginates_until_no_next_url(self, mock_client_cls: MagicMock) -> None:
        """Must follow pagination until next_page_url is None."""
        doc1 = {"document_number": "2024-00001", "title": "Rule 1", "type": "Rule"}
        doc2 = {"document_number": "2024-00002", "title": "Rule 2", "type": "Rule"}

        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        # First call returns doc1 + next_url; second call returns doc2 + no next_url
        mock_client.get.side_effect = [
            self._make_mock_response([doc1], next_url="http://example.com/page=2"),
            self._make_mock_response([doc2], next_url=None),
        ]

        results = fetch_federal_register(since_date="2024-01-01")

        assert len(results) == 2
        assert mock_client.get.call_count == 2

    @patch("backend.pipelines.ingestion.httpx.Client")
    def test_empty_response_returns_empty_list(self, mock_client_cls: MagicMock) -> None:
        """Empty API response should return empty list, not raise."""
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.get.return_value = self._make_mock_response([])

        results = fetch_federal_register(since_date="2024-01-01")

        assert results == []
