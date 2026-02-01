# tests/test_gopls_client.py
"""Tests for gopls LSP client."""

import pytest
import tempfile
import os
from pathlib import Path


class TestGoplsClientInit:
    """Test GoplsClient initialization."""

    def test_gopls_available_check(self):
        """GOPLS_AVAILABLE should reflect whether gopls is in PATH."""
        from tldr.gopls_client import GOPLS_AVAILABLE
        import shutil

        expected = shutil.which("gopls") is not None
        assert GOPLS_AVAILABLE == expected

    def test_client_creation(self):
        """Client can be created with project root."""
        from tldr.gopls_client import GoplsClient

        with tempfile.TemporaryDirectory() as tmpdir:
            client = GoplsClient(project_root=tmpdir)
            assert client.project_root == Path(tmpdir).resolve()
            assert client._process is None
            assert client._initialized is False
