# tests/test_gopls_client.py
"""Tests for gopls LSP client."""

import pytest
import tempfile
import os
import shutil
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


class TestGoplsClientLifecycle:
    """Test gopls subprocess lifecycle."""

    @pytest.fixture
    def go_project(self):
        """Create a minimal Go project for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create go.mod
            go_mod = Path(tmpdir) / "go.mod"
            go_mod.write_text("module testproject\n\ngo 1.21\n")

            # Create a simple Go file
            main_go = Path(tmpdir) / "main.go"
            main_go.write_text('''package main

func main() {
    println("hello")
}
''')
            yield tmpdir

    @pytest.mark.skipif(
        not shutil.which("gopls"),
        reason="gopls not installed"
    )
    def test_client_start_stop(self, go_project):
        """Client can start and stop gopls subprocess."""
        from tldr.gopls_client import GoplsClient

        client = GoplsClient(project_root=go_project)

        # Start
        assert client.start() is True
        assert client.is_running() is True
        assert client._initialized is True

        # Stop
        client.stop()
        assert client.is_running() is False
        assert client._initialized is False

    def test_client_start_without_gopls(self, monkeypatch):
        """Client returns False if gopls not available."""
        from tldr import gopls_client
        from tldr.gopls_client import GoplsClient

        # Pretend gopls is not available
        monkeypatch.setattr(gopls_client, "GOPLS_AVAILABLE", False)

        with tempfile.TemporaryDirectory() as tmpdir:
            client = GoplsClient(project_root=tmpdir)
            assert client.start() is False


class TestGoplsHover:
    """Test gopls hover (signature extraction)."""

    @pytest.fixture
    def go_project_with_func(self):
        """Create a Go project with a typed function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            go_mod = Path(tmpdir) / "go.mod"
            go_mod.write_text("module testproject\n\ngo 1.21\n")

            main_go = Path(tmpdir) / "main.go"
            main_go.write_text('''package main

import "context"

// ProcessData handles data processing with context.
func ProcessData(ctx context.Context, data []byte) (int, error) {
    return len(data), nil
}

func main() {
    ProcessData(context.Background(), nil)
}
''')
            yield tmpdir

    @pytest.mark.skipif(
        not shutil.which("gopls"),
        reason="gopls not installed"
    )
    def test_get_symbol_at(self, go_project_with_func):
        """Can get symbol info with typed signature."""
        from tldr.gopls_client import GoplsClient

        client = GoplsClient(project_root=go_project_with_func)
        assert client.start()

        try:
            main_go = str(Path(go_project_with_func) / "main.go")
            # Line 5 (0-indexed) is where ProcessData is defined
            symbol = client.get_symbol_at(main_go, line=5, col=5)

            assert symbol is not None
            assert symbol.name == "ProcessData"
            assert "context.Context" in symbol.signature
            assert "[]byte" in symbol.signature
            assert "(int, error)" in symbol.signature
            assert "ProcessData handles" in symbol.doc
        finally:
            client.stop()


class TestGoplsCallHierarchy:
    """Test gopls call hierarchy.

    Note: gopls call hierarchy works reliably within a single file.
    Cross-file call hierarchy may require more workspace initialization
    time or additional LSP configuration in some environments.
    """

    @pytest.fixture
    def go_project_with_calls(self):
        """Create a Go project with function calls.

        Uses a single file to ensure reliable call hierarchy results.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            go_mod = Path(tmpdir) / "go.mod"
            go_mod.write_text("module testproject\n\ngo 1.21\n")

            # Single file with main() calling Helper()
            # Line numbers (0-indexed):
            # 0: package main
            # 1: (empty)
            # 2: func main() {
            # 3:     result := Helper()
            # 4:     println(result)
            # 5: }
            # 6: (empty)
            # 7: // Helper returns a greeting.
            # 8: func Helper() string {
            # 9:     return "hello"
            # 10: }
            main_go = Path(tmpdir) / "main.go"
            main_go.write_text('''package main

func main() {
    result := Helper()
    println(result)
}

// Helper returns a greeting.
func Helper() string {
    return "hello"
}
''')
            yield tmpdir

    @pytest.mark.skipif(
        not shutil.which("gopls"),
        reason="gopls not installed"
    )
    def test_get_outgoing_calls(self, go_project_with_calls):
        """Can get outgoing calls from a function."""
        import time
        from tldr.gopls_client import GoplsClient

        client = GoplsClient(project_root=go_project_with_calls)
        assert client.start()

        try:
            main_go = str(Path(go_project_with_calls) / "main.go")
            # Give gopls time to index
            time.sleep(2)
            # Line 2 (0-indexed) is where main() is defined
            calls = client.get_outgoing_calls(main_go, line=2, col=5)

            # main() calls Helper()
            call_names = [c.to_func for c in calls]
            assert "Helper" in call_names
        finally:
            client.stop()

    @pytest.mark.skipif(
        not shutil.which("gopls"),
        reason="gopls not installed"
    )
    def test_get_incoming_calls(self, go_project_with_calls):
        """Can get incoming calls to a function."""
        import time
        from tldr.gopls_client import GoplsClient

        client = GoplsClient(project_root=go_project_with_calls)
        assert client.start()

        try:
            main_go = str(Path(go_project_with_calls) / "main.go")
            # Give gopls time to index
            time.sleep(2)
            # Line 8 (0-indexed) is where Helper() is defined
            calls = client.get_incoming_calls(main_go, line=8, col=5)

            # Helper() is called by main()
            caller_names = [c.from_func for c in calls]
            assert "main" in caller_names
        finally:
            client.stop()
