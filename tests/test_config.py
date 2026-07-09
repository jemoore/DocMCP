from __future__ import annotations

import os
from pathlib import Path

import pytest

from docmcp.config import Config


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test from a clean slate so the ambient shell environment
    cannot leak DOCMCP_* settings into from_env()."""
    for key in list(os.environ):
        if key.startswith("DOCMCP_"):
            monkeypatch.delenv(key, raising=False)


class TestConfigFromEnv:
    def test_defaults(self) -> None:
        config = Config.from_env()
        assert config.port == 8808
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.auth_token is None

    def test_auth_token_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCMCP_AUTH_TOKEN", "secret123")
        assert Config.from_env().auth_token == "secret123"

    def test_blank_auth_token_becomes_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOCMCP_AUTH_TOKEN", "   ")
        assert Config.from_env().auth_token is None

    def test_multiple_doc_dirs_parsed_and_trimmed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOCMCP_DOC_DIRS", "/a, /b ,/c")
        assert Config.from_env().doc_dirs == [Path("/a"), Path("/b"), Path("/c")]

    def test_overlap_must_be_less_than_chunk_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOCMCP_CHUNK_SIZE", "500")
        monkeypatch.setenv("DOCMCP_CHUNK_OVERLAP", "500")
        with pytest.raises(ValueError, match="must be smaller"):
            Config.from_env()
