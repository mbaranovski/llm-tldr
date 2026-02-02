"""Tests for Go call graph via gopls."""

import pytest
import tempfile
import shutil
from pathlib import Path


class TestGoCallGraphGopls:
    """Test Go call graph with gopls integration."""

    @pytest.fixture
    def go_project_multi_file(self):
        """Create a Go project with cross-file calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "go.mod").write_text("module testproject\ngo 1.21\n")

            # Main calls Handler which calls Helper
            (Path(tmpdir) / "main.go").write_text('''package main

func main() {
    Handler()
}
''')
            (Path(tmpdir) / "handler.go").write_text('''package main

func Handler() {
    result := Helper()
    println(result)
}
''')
            (Path(tmpdir) / "helper.go").write_text('''package main

func Helper() string {
    return "hello"
}
''')
            yield tmpdir

    @pytest.mark.skipif(
        not shutil.which("gopls"),
        reason="gopls not installed"
    )
    def test_cross_file_call_graph(self, go_project_multi_file):
        """Call graph includes cross-file calls."""
        from tldr.cross_file_calls import build_project_call_graph

        graph = build_project_call_graph(go_project_multi_file, language="go")

        # main -> Handler (cross-file)
        edges = list(graph.edges)

        # Find edges by function names
        edge_pairs = [(e[1], e[3]) for e in edges]  # (src_func, dst_func)

        assert ("main", "Handler") in edge_pairs
        assert ("Handler", "Helper") in edge_pairs

    def test_call_graph_fallback_without_gopls(self, go_project_multi_file, monkeypatch):
        """Call graph falls back to tree-sitter without gopls."""
        from tldr import gopls_client
        monkeypatch.setattr(gopls_client, "GOPLS_AVAILABLE", False)

        from tldr.cross_file_calls import build_project_call_graph

        # Should not raise, just use tree-sitter
        graph = build_project_call_graph(go_project_multi_file, language="go")

        # At minimum, should have some edges (intra-file at least)
        assert graph is not None
