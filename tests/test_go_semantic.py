"""Tests for Go semantic integration."""

import pytest
import tempfile
import shutil
from pathlib import Path


class TestGoSignatureExtraction:
    """Test Go signature extraction via gopls."""

    @pytest.fixture
    def go_project(self):
        """Create a Go project with typed functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "go.mod").write_text("module testproject\ngo 1.21\n")

            (Path(tmpdir) / "main.go").write_text('''package main

import "context"

// ProcessData processes incoming data.
func ProcessData(ctx context.Context, data []byte) (int, error) {
    return len(data), nil
}
''')
            yield tmpdir

    @pytest.mark.skipif(
        not shutil.which("gopls"),
        reason="gopls not installed"
    )
    def test_go_signature_with_gopls(self, go_project):
        """Go signature includes full types when gopls available."""
        from tldr.semantic import _get_function_signature

        main_go = Path(go_project) / "main.go"
        sig = _get_function_signature(main_go, "ProcessData", "go")

        assert sig is not None
        assert "context.Context" in sig
        assert "[]byte" in sig
        assert "int" in sig
        assert "error" in sig

    def test_go_signature_fallback(self, go_project, monkeypatch):
        """Go signature falls back gracefully without gopls."""
        from tldr import gopls_client
        monkeypatch.setattr(gopls_client, "GOPLS_AVAILABLE", False)

        from tldr.semantic import _get_function_signature

        main_go = Path(go_project) / "main.go"
        sig = _get_function_signature(main_go, "ProcessData", "go")

        # Should return basic signature
        assert sig is not None
        assert "ProcessData" in sig


class TestGoDocstringExtraction:
    """Test Go docstring extraction via gopls."""

    @pytest.fixture
    def go_project(self):
        """Create a Go project with documented functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "go.mod").write_text("module testproject\ngo 1.21\n")

            (Path(tmpdir) / "main.go").write_text('''package main

// ProcessData processes incoming data and returns the count.
// It handles nil data gracefully.
func ProcessData(data []byte) int {
    if data == nil {
        return 0
    }
    return len(data)
}
''')
            yield tmpdir

    @pytest.mark.skipif(
        not shutil.which("gopls"),
        reason="gopls not installed"
    )
    def test_go_docstring_with_gopls(self, go_project):
        """Go docstring extracted when gopls available."""
        from tldr.semantic import _get_function_docstring

        main_go = Path(go_project) / "main.go"
        doc = _get_function_docstring(main_go, "ProcessData", "go")

        assert doc is not None
        assert "processes incoming data" in doc.lower()
