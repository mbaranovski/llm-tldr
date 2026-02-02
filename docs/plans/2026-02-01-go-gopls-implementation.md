# Go gopls Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade Go from "Basic" to "Full" semantic support by integrating gopls for type-aware signatures and call graphs.

**Architecture:** Add a GoplsClient that manages gopls subprocess lifecycle and JSON-RPC communication. Integrate with semantic.py for signatures/docstrings and cross_file_calls.py for type-aware call graphs. Tree-sitter remains as fallback when gopls is unavailable.

**Tech Stack:** Python 3.12, gopls (Go Language Server), JSON-RPC 2.0, LSP protocol, tree-sitter (fallback)

---

## Task 1: Create GoplsClient Core

**Files:**
- Create: `tldr/gopls_client.py`
- Create: `tests/test_gopls_client.py`

**Step 1: Write the failing test for client initialization**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_gopls_client.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'tldr.gopls_client'"

**Step 3: Write minimal implementation**

```python
# tldr/gopls_client.py
"""
gopls Language Server Protocol client for Go semantic analysis.

Provides:
- Subprocess lifecycle management (start/stop gopls)
- JSON-RPC 2.0 communication over stdin/stdout
- LSP methods: hover, callHierarchy, references
- Caching layer to avoid repeated requests
- Graceful fallback when gopls is unavailable
"""

import json
import subprocess
import threading
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
import shutil

logger = logging.getLogger("tldr.gopls")

# Check if gopls is available
GOPLS_AVAILABLE = shutil.which("gopls") is not None


@dataclass
class GoSymbol:
    """Resolved Go symbol with type information."""
    name: str
    qualified_name: str
    signature: str
    doc: str
    file: str
    line: int
    kind: str
    receiver: Optional[str] = None


@dataclass
class GoCallEdge:
    """A resolved call graph edge."""
    from_file: str
    from_func: str
    from_line: int
    to_file: str
    to_func: str
    to_line: int


@dataclass
class GoplsClient:
    """LSP client for gopls."""
    project_root: Path
    _process: Optional[subprocess.Popen] = field(default=None, repr=False)
    _request_id: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _initialized: bool = field(default=False, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.project_root = Path(self.project_root).resolve()

    def is_running(self) -> bool:
        """Check if gopls is running."""
        return self._process is not None and self._process.poll() is None
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_gopls_client.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add tldr/gopls_client.py tests/test_gopls_client.py
git commit -m "feat(gopls): add GoplsClient core with dataclasses

- Add GOPLS_AVAILABLE check
- Add GoSymbol and GoCallEdge dataclasses
- Add GoplsClient with project_root initialization

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add gopls Start/Stop Lifecycle

**Files:**
- Modify: `tldr/gopls_client.py`
- Modify: `tests/test_gopls_client.py`

**Step 1: Write the failing test for start/stop**

```python
# Add to tests/test_gopls_client.py

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
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_gopls_client.py::TestGoplsClientLifecycle -v`
Expected: FAIL with "AttributeError: 'GoplsClient' object has no attribute 'start'"

**Step 3: Write minimal implementation**

```python
# Add to GoplsClient class in tldr/gopls_client.py

    def start(self) -> bool:
        """Start gopls subprocess. Returns True if successful."""
        if not GOPLS_AVAILABLE:
            logger.warning("gopls not found in PATH")
            return False

        if self._process is not None:
            return True

        try:
            self._process = subprocess.Popen(
                ["gopls", "serve"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.project_root),
            )

            # Initialize LSP connection
            self._send_request("initialize", {
                "processId": None,
                "rootUri": f"file://{self.project_root}",
                "capabilities": {
                    "textDocument": {
                        "hover": {"contentFormat": ["plaintext", "markdown"]},
                        "callHierarchy": {"dynamicRegistration": False},
                    }
                },
            })

            self._send_notification("initialized", {})
            self._initialized = True
            logger.info(f"gopls started for {self.project_root}")
            return True

        except Exception as e:
            logger.error(f"Failed to start gopls: {e}")
            self._process = None
            return False

    def stop(self):
        """Stop gopls subprocess."""
        if self._process:
            try:
                self._send_request("shutdown", {})
                self._send_notification("exit", {})
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            finally:
                self._process = None
                self._initialized = False
                logger.info("gopls stopped")

    def _send_request(self, method: str, params: dict) -> Any:
        """Send JSON-RPC request and wait for response."""
        if not self.is_running():
            return None

        with self._lock:
            self._request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            }

            try:
                content = json.dumps(request)
                message = f"Content-Length: {len(content)}\r\n\r\n{content}"
                self._process.stdin.write(message.encode())
                self._process.stdin.flush()

                response = self._read_response()

                if "error" in response:
                    logger.warning(f"gopls error: {response['error']}")
                    return None

                return response.get("result")

            except Exception as e:
                logger.error(f"gopls request failed: {e}")
                return None

    def _send_notification(self, method: str, params: dict):
        """Send JSON-RPC notification (no response expected)."""
        if not self.is_running():
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        try:
            content = json.dumps(notification)
            message = f"Content-Length: {len(content)}\r\n\r\n{content}"
            self._process.stdin.write(message.encode())
            self._process.stdin.flush()
        except Exception as e:
            logger.warning(f"gopls notification failed: {e}")

    def _read_response(self) -> dict:
        """Read JSON-RPC response from gopls stdout."""
        headers = {}
        while True:
            line = self._process.stdout.readline().decode().strip()
            if not line:
                break
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip()] = value.strip()

        content_length = int(headers.get("Content-Length", 0))
        if content_length > 0:
            content = self._process.stdout.read(content_length).decode()
            return json.loads(content)

        return {}
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_gopls_client.py::TestGoplsClientLifecycle -v`
Expected: PASS (2 tests, 1 possibly skipped if no gopls)

**Step 5: Commit**

```bash
git add tldr/gopls_client.py tests/test_gopls_client.py
git commit -m "feat(gopls): add start/stop lifecycle management

- Add start() with LSP initialize handshake
- Add stop() with graceful shutdown
- Add JSON-RPC request/notification/response handling
- Add tests for lifecycle (skipped if no gopls)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Add Hover (Signature) Support

**Files:**
- Modify: `tldr/gopls_client.py`
- Modify: `tests/test_gopls_client.py`

**Step 1: Write the failing test for hover**

```python
# Add to tests/test_gopls_client.py
import shutil

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
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_gopls_client.py::TestGoplsHover -v`
Expected: FAIL with "AttributeError: 'GoplsClient' object has no attribute 'get_symbol_at'"

**Step 3: Write minimal implementation**

```python
# Add to GoplsClient class in tldr/gopls_client.py

    def get_symbol_at(self, file_path: str, line: int, col: int) -> Optional[GoSymbol]:
        """
        Get symbol information at a position using textDocument/hover.

        Args:
            file_path: Absolute path to Go file
            line: 0-indexed line number
            col: 0-indexed column number

        Returns:
            GoSymbol with signature and documentation, or None
        """
        cache_key = ("hover", file_path, line, col)
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._ensure_file_open(file_path)

        result = self._send_request("textDocument/hover", {
            "textDocument": {"uri": f"file://{file_path}"},
            "position": {"line": line, "character": col},
        })

        if not result or "contents" not in result:
            return None

        symbol = self._parse_hover_result(result, file_path, line)
        self._cache[cache_key] = symbol
        return symbol

    def _ensure_file_open(self, file_path: str):
        """Notify gopls that a file is open."""
        cache_key = ("open", file_path)
        if cache_key in self._cache:
            return

        try:
            content = Path(file_path).read_text()
            self._send_notification("textDocument/didOpen", {
                "textDocument": {
                    "uri": f"file://{file_path}",
                    "languageId": "go",
                    "version": 1,
                    "text": content,
                }
            })
            self._cache[cache_key] = True
        except Exception as e:
            logger.warning(f"Failed to open {file_path}: {e}")

    def _parse_hover_result(self, result: dict, file_path: str, line: int) -> Optional[GoSymbol]:
        """Parse hover result into GoSymbol."""
        contents = result.get("contents", {})

        if isinstance(contents, dict):
            value = contents.get("value", "")
        elif isinstance(contents, str):
            value = contents
        else:
            return None

        lines = value.strip().split("\n")
        signature = ""
        doc = ""
        in_code_block = False

        for l in lines:
            if l.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                signature = l.strip()
            else:
                if doc:
                    doc += "\n"
                doc += l

        if not signature:
            return None

        name, receiver, kind = self._parse_go_signature(signature)

        return GoSymbol(
            name=name,
            qualified_name=f"({receiver}).{name}" if receiver else name,
            signature=signature,
            doc=doc.strip(),
            file=file_path,
            line=line,
            kind=kind,
            receiver=receiver,
        )

    def _parse_go_signature(self, sig: str) -> tuple[str, Optional[str], str]:
        """Parse Go signature to extract name, receiver, kind."""
        sig = sig.strip()

        if sig.startswith("type "):
            parts = sig[5:].split()
            name = parts[0] if parts else ""
            kind = "interface" if "interface" in sig else "type"
            return (name, None, kind)

        if sig.startswith("func "):
            rest = sig[5:].strip()

            if rest.startswith("("):
                depth = 0
                end = 0
                for i, c in enumerate(rest):
                    if c == "(":
                        depth += 1
                    elif c == ")":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break

                receiver_part = rest[1:end-1].strip()
                receiver_parts = receiver_part.split()
                receiver = receiver_parts[-1] if receiver_parts else None

                rest = rest[end:].strip()
                name_end = rest.find("(")
                name = rest[:name_end].strip() if name_end > 0 else rest

                return (name, receiver, "method")
            else:
                name_end = rest.find("(")
                name = rest[:name_end].strip() if name_end > 0 else rest
                return (name, None, "function")

        return ("", None, "unknown")
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_gopls_client.py::TestGoplsHover -v`
Expected: PASS (1 test, possibly skipped if no gopls)

**Step 5: Commit**

```bash
git add tldr/gopls_client.py tests/test_gopls_client.py
git commit -m "feat(gopls): add hover support for typed signatures

- Add get_symbol_at() using textDocument/hover
- Add _ensure_file_open() for didOpen notification
- Add _parse_hover_result() to extract signature/doc
- Add _parse_go_signature() to parse func signatures
- Add caching for hover results

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Add Call Hierarchy Support

**Files:**
- Modify: `tldr/gopls_client.py`
- Modify: `tests/test_gopls_client.py`

**Step 1: Write the failing test for call hierarchy**

```python
# Add to tests/test_gopls_client.py

class TestGoplsCallHierarchy:
    """Test gopls call hierarchy."""

    @pytest.fixture
    def go_project_with_calls(self):
        """Create a Go project with cross-file calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            go_mod = Path(tmpdir) / "go.mod"
            go_mod.write_text("module testproject\n\ngo 1.21\n")

            # Main file that calls helper
            main_go = Path(tmpdir) / "main.go"
            main_go.write_text('''package main

func main() {
    result := Helper()
    println(result)
}
''')

            # Helper file
            helper_go = Path(tmpdir) / "helper.go"
            helper_go.write_text('''package main

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
        from tldr.gopls_client import GoplsClient

        client = GoplsClient(project_root=go_project_with_calls)
        assert client.start()

        try:
            main_go = str(Path(go_project_with_calls) / "main.go")
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
        from tldr.gopls_client import GoplsClient

        client = GoplsClient(project_root=go_project_with_calls)
        assert client.start()

        try:
            helper_go = str(Path(go_project_with_calls) / "helper.go")
            # Line 3 (0-indexed) is where Helper() is defined
            calls = client.get_incoming_calls(helper_go, line=3, col=5)

            # Helper() is called by main()
            caller_names = [c.from_func for c in calls]
            assert "main" in caller_names
        finally:
            client.stop()
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_gopls_client.py::TestGoplsCallHierarchy -v`
Expected: FAIL with "AttributeError: 'GoplsClient' object has no attribute 'get_outgoing_calls'"

**Step 3: Write minimal implementation**

```python
# Add to GoplsClient class in tldr/gopls_client.py

    def get_outgoing_calls(self, file_path: str, line: int, col: int) -> list[GoCallEdge]:
        """Get functions called by the function at position."""
        cache_key = ("outgoing", file_path, line, col)
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._ensure_file_open(file_path)

        items = self._send_request("textDocument/prepareCallHierarchy", {
            "textDocument": {"uri": f"file://{file_path}"},
            "position": {"line": line, "character": col},
        })

        if not items:
            return []

        edges = []
        for item in items:
            outgoing = self._send_request("callHierarchy/outgoingCalls", {
                "item": item
            })

            if outgoing:
                for call in outgoing:
                    edge = self._parse_call_edge(item, call["to"], "outgoing")
                    if edge:
                        edges.append(edge)

        self._cache[cache_key] = edges
        return edges

    def get_incoming_calls(self, file_path: str, line: int, col: int) -> list[GoCallEdge]:
        """Get functions that call the function at position."""
        cache_key = ("incoming", file_path, line, col)
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._ensure_file_open(file_path)

        items = self._send_request("textDocument/prepareCallHierarchy", {
            "textDocument": {"uri": f"file://{file_path}"},
            "position": {"line": line, "character": col},
        })

        if not items:
            return []

        edges = []
        for item in items:
            incoming = self._send_request("callHierarchy/incomingCalls", {
                "item": item
            })

            if incoming:
                for call in incoming:
                    edge = self._parse_call_edge(call["from"], item, "incoming")
                    if edge:
                        edges.append(edge)

        self._cache[cache_key] = edges
        return edges

    def _parse_call_edge(self, from_item: dict, to_item: dict, direction: str) -> Optional[GoCallEdge]:
        """Parse call hierarchy items into GoCallEdge."""
        try:
            from_uri = from_item.get("uri", "")
            to_uri = to_item.get("uri", "")

            from_file = from_uri[7:] if from_uri.startswith("file://") else from_uri
            to_file = to_uri[7:] if to_uri.startswith("file://") else to_uri

            return GoCallEdge(
                from_file=from_file,
                from_func=from_item.get("name", ""),
                from_line=from_item.get("range", {}).get("start", {}).get("line", 0),
                to_file=to_file,
                to_func=to_item.get("name", ""),
                to_line=to_item.get("range", {}).get("start", {}).get("line", 0),
            )
        except Exception:
            return None
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_gopls_client.py::TestGoplsCallHierarchy -v`
Expected: PASS (2 tests, possibly skipped if no gopls)

**Step 5: Commit**

```bash
git add tldr/gopls_client.py tests/test_gopls_client.py
git commit -m "feat(gopls): add call hierarchy support

- Add get_outgoing_calls() using callHierarchy/outgoingCalls
- Add get_incoming_calls() using callHierarchy/incomingCalls
- Add _parse_call_edge() to convert LSP items to GoCallEdge
- Add caching for call hierarchy results

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Add Module-Level Client Management

**Files:**
- Modify: `tldr/gopls_client.py`
- Modify: `tests/test_gopls_client.py`

**Step 1: Write the failing test for singleton management**

```python
# Add to tests/test_gopls_client.py

class TestGoplsClientManager:
    """Test module-level client management."""

    def test_get_gopls_client_returns_none_without_gopls(self, monkeypatch):
        """get_gopls_client returns None if gopls unavailable."""
        from tldr import gopls_client
        from tldr.gopls_client import get_gopls_client

        monkeypatch.setattr(gopls_client, "GOPLS_AVAILABLE", False)

        with tempfile.TemporaryDirectory() as tmpdir:
            client = get_gopls_client(tmpdir)
            assert client is None

    @pytest.mark.skipif(
        not shutil.which("gopls"),
        reason="gopls not installed"
    )
    def test_get_gopls_client_singleton(self):
        """get_gopls_client returns same instance for same project."""
        from tldr.gopls_client import get_gopls_client, shutdown_all_clients

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create go.mod
            (Path(tmpdir) / "go.mod").write_text("module test\ngo 1.21\n")

            try:
                client1 = get_gopls_client(tmpdir)
                client2 = get_gopls_client(tmpdir)

                assert client1 is client2
            finally:
                shutdown_all_clients()

    def test_shutdown_all_clients(self):
        """shutdown_all_clients clears all clients."""
        from tldr.gopls_client import shutdown_all_clients, _clients

        shutdown_all_clients()
        assert len(_clients) == 0
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_gopls_client.py::TestGoplsClientManager -v`
Expected: FAIL with "ImportError: cannot import name 'get_gopls_client'"

**Step 3: Write minimal implementation**

```python
# Add at end of tldr/gopls_client.py

# === Module-level singleton for reuse ===

_clients: dict[str, GoplsClient] = {}
_clients_lock = threading.Lock()


def get_gopls_client(project_root: str | Path) -> Optional[GoplsClient]:
    """
    Get or create a GoplsClient for a project.

    Returns None if gopls is not available.
    """
    if not GOPLS_AVAILABLE:
        return None

    project_root = str(Path(project_root).resolve())

    with _clients_lock:
        if project_root not in _clients:
            client = GoplsClient(project_root=project_root)
            if client.start():
                _clients[project_root] = client
            else:
                return None

        return _clients[project_root]


def shutdown_all_clients():
    """Shutdown all gopls clients. Call on daemon shutdown."""
    with _clients_lock:
        for client in _clients.values():
            client.stop()
        _clients.clear()
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_gopls_client.py::TestGoplsClientManager -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add tldr/gopls_client.py tests/test_gopls_client.py
git commit -m "feat(gopls): add module-level client management

- Add get_gopls_client() singleton factory
- Add shutdown_all_clients() for cleanup
- Add _clients dict with thread-safe access

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Integrate with semantic.py for Signatures

**Files:**
- Modify: `tldr/semantic.py`
- Create: `tests/test_go_semantic.py`

**Step 1: Write the failing test**

```python
# tests/test_go_semantic.py
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
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_go_semantic.py -v`
Expected: FAIL (signature doesn't include types, returns "function ProcessData(...)")

**Step 3: Write minimal implementation**

```python
# Modify tldr/semantic.py

# Add import at top (after existing imports)
from tldr.gopls_client import get_gopls_client, GOPLS_AVAILABLE


# Modify _get_function_signature function - replace the existing one:

def _get_function_signature(file_path: Path, func_name: str, lang: str) -> Optional[str]:
    """Extract function signature from file."""
    if not file_path.exists():
        return None

    try:
        content = file_path.read_text()

        if lang == "python":
            import ast
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    args = []
                    for arg in node.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        args.append(arg_str)

                    returns = ""
                    if node.returns:
                        returns = f" -> {ast.unparse(node.returns)}"

                    return f"def {func_name}({', '.join(args)}){returns}"

        elif lang == "go" and GOPLS_AVAILABLE:
            signature = _get_go_signature_via_gopls(file_path, func_name, content)
            if signature:
                return signature

        # Fallback for all other languages
        return f"function {func_name}(...)"

    except Exception:
        return None


def _get_go_signature_via_gopls(file_path: Path, func_name: str, content: str) -> Optional[str]:
    """Get Go function signature using gopls."""
    project_root = _find_go_module_root(file_path)
    if not project_root:
        return None

    client = get_gopls_client(project_root)
    if not client:
        return None

    func_line = _find_go_func_line(content, func_name)
    if func_line is None:
        return None

    symbol = client.get_symbol_at(str(file_path.resolve()), func_line, 0)
    if symbol and symbol.signature:
        return symbol.signature

    return None


def _find_go_module_root(file_path: Path) -> Optional[Path]:
    """Find the nearest directory containing go.mod."""
    current = file_path.parent.resolve()
    while current != current.parent:
        if (current / "go.mod").exists():
            return current
        current = current.parent
    return None


def _find_go_func_line(content: str, func_name: str) -> Optional[int]:
    """Find the line number where a Go function is defined."""
    import re

    func_pattern = rf'^\s*func\s+{re.escape(func_name)}\s*\('
    method_pattern = rf'^\s*func\s+\([^)]+\)\s+{re.escape(func_name)}\s*\('

    for i, line in enumerate(content.split('\n')):
        if re.search(func_pattern, line) or re.search(method_pattern, line):
            return i

    return None
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_go_semantic.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add tldr/semantic.py tests/test_go_semantic.py
git commit -m "feat(semantic): integrate gopls for Go signatures

- Add gopls integration to _get_function_signature()
- Add _get_go_signature_via_gopls() helper
- Add _find_go_module_root() to locate go.mod
- Add _find_go_func_line() to find function definitions
- Graceful fallback when gopls unavailable

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Integrate with semantic.py for Docstrings

**Files:**
- Modify: `tldr/semantic.py`
- Modify: `tests/test_go_semantic.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_go_semantic.py

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
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_go_semantic.py::TestGoDocstringExtraction -v`
Expected: FAIL (returns None for Go)

**Step 3: Write minimal implementation**

```python
# Modify _get_function_docstring in tldr/semantic.py

def _get_function_docstring(file_path: Path, func_name: str, lang: str) -> Optional[str]:
    """Extract function docstring from file."""
    if not file_path.exists():
        return None

    try:
        content = file_path.read_text()

        if lang == "python":
            import ast
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    return ast.get_docstring(node)

        elif lang == "go" and GOPLS_AVAILABLE:
            docstring = _get_go_docstring_via_gopls(file_path, func_name, content)
            if docstring:
                return docstring

        return None

    except Exception:
        return None


def _get_go_docstring_via_gopls(file_path: Path, func_name: str, content: str) -> Optional[str]:
    """Get Go function documentation using gopls."""
    project_root = _find_go_module_root(file_path)
    if not project_root:
        return None

    client = get_gopls_client(project_root)
    if not client:
        return None

    func_line = _find_go_func_line(content, func_name)
    if func_line is None:
        return None

    symbol = client.get_symbol_at(str(file_path.resolve()), func_line, 0)
    if symbol and symbol.doc:
        return symbol.doc

    return None
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_go_semantic.py::TestGoDocstringExtraction -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tldr/semantic.py tests/test_go_semantic.py
git commit -m "feat(semantic): integrate gopls for Go docstrings

- Add gopls integration to _get_function_docstring()
- Add _get_go_docstring_via_gopls() helper
- Extract Go doc comments via hover

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Integrate with cross_file_calls.py

**Files:**
- Modify: `tldr/cross_file_calls.py`
- Create: `tests/test_go_call_graph.py`

**Step 1: Write the failing test**

```python
# tests/test_go_call_graph.py
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
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_go_call_graph.py -v`
Expected: FAIL (cross-file calls not resolved)

**Step 3: Write minimal implementation**

```python
# Modify tldr/cross_file_calls.py

# Add import at top
from tldr.gopls_client import get_gopls_client, GOPLS_AVAILABLE


# Replace _build_go_call_graph function:

def _build_go_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    func_index: dict,
    workspace_config: Optional[WorkspaceConfig] = None
):
    """Build call graph for Go files."""

    # Try gopls first (type-aware, accurate)
    if GOPLS_AVAILABLE and (root / "go.mod").exists():
        success = _build_go_call_graph_gopls(root, graph, workspace_config)
        if success:
            return

    # Fallback: tree-sitter based (existing code, limited)
    _build_go_call_graph_treesitter(root, graph, func_index, workspace_config)


def _build_go_call_graph_gopls(
    root: Path,
    graph: ProjectCallGraph,
    workspace_config: Optional[WorkspaceConfig] = None
) -> bool:
    """Build Go call graph using gopls call hierarchy."""
    client = get_gopls_client(root)
    if not client:
        return False

    go_files = scan_project(root, "go", workspace_config)
    if not go_files:
        return False

    for go_file in go_files:
        go_path = Path(go_file)
        rel_path = str(go_path.relative_to(root))

        try:
            content = go_path.read_text()
        except Exception:
            continue

        func_locations = _find_all_go_funcs(content)

        for func_name, line in func_locations:
            try:
                outgoing = client.get_outgoing_calls(str(go_path.resolve()), line, 0)

                for edge in outgoing:
                    try:
                        to_rel = str(Path(edge.to_file).relative_to(root))
                    except ValueError:
                        continue

                    graph.add_edge(rel_path, func_name, to_rel, edge.to_func)

            except Exception:
                pass

    return True


def _find_all_go_funcs(content: str) -> list[tuple[str, int]]:
    """Find all function/method definitions in Go source."""
    import re

    results = []
    lines = content.split('\n')

    func_pattern = re.compile(r'^\s*func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(')
    method_pattern = re.compile(r'^\s*func\s+\([^)]+\)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(')

    for i, line in enumerate(lines):
        match = func_pattern.search(line)
        if match:
            results.append((match.group(1), i))
            continue

        match = method_pattern.search(line)
        if match:
            results.append((match.group(1), i))

    return results


def _build_go_call_graph_treesitter(
    root: Path,
    graph: ProjectCallGraph,
    func_index: dict,
    workspace_config: Optional[WorkspaceConfig] = None
):
    """Fallback: Build Go call graph using tree-sitter."""
    # Keep existing implementation - rename from _build_go_call_graph
    for go_file in scan_project(root, "go", workspace_config):
        go_path = Path(go_file)
        rel_path = str(go_path.relative_to(root))

        imports = parse_go_imports(go_path)
        package_imports = {}

        for imp in imports:
            module = imp['module']
            alias = imp.get('alias')

            if module.startswith('./') or module.startswith('../'):
                module_path = _resolve_go_import(rel_path, module)
            else:
                module_path = module

            if alias:
                local_name = alias
            else:
                local_name = module.rstrip('/').split('/')[-1]

            package_imports[local_name] = module_path

        calls_by_func = _extract_go_file_calls(go_path, root)

        for caller_func, calls in calls_by_func.items():
            for call_type, call_target in calls:
                if call_type == 'intra':
                    graph.add_edge(rel_path, caller_func, rel_path, call_target)

                elif call_type == 'attr':
                    parts = call_target.split('.', 1)
                    if len(parts) == 2:
                        pkg, func_name = parts
                        if pkg in package_imports:
                            pkg_path = package_imports[pkg]
                            for key, file_path in func_index.items():
                                if isinstance(key, tuple) and len(key) == 2:
                                    mod, name = key
                                    if name == func_name:
                                        if pkg_path.lstrip('./') in file_path or mod == pkg:
                                            graph.add_edge(rel_path, caller_func, file_path, func_name)
                                            break
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_go_call_graph.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add tldr/cross_file_calls.py tests/test_go_call_graph.py
git commit -m "feat(call-graph): integrate gopls for Go call graph

- Add _build_go_call_graph_gopls() using callHierarchy
- Add _find_all_go_funcs() to locate function definitions
- Rename existing impl to _build_go_call_graph_treesitter()
- Auto-select gopls if available, fallback to tree-sitter

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Update Documentation

**Files:**
- Modify: `docs/TLDR.md`

**Step 1: No test needed for docs**

**Step 2: N/A**

**Step 3: Update the table in docs/TLDR.md**

Find the table around line 200 and change Go from "⚠️ Basic" to "✅ Full":

```markdown
| Language | AST | Call Graph | CFG | DFG | PDG | Semantic* |
|----------|-----|------------|-----|-----|-----|-----------|
| Python | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ Full |
| TypeScript | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ Full |
| JavaScript | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ Full |
| Go | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ Full |
```

Also add a note about gopls requirement:

```markdown
> **Note:** Full Go semantic support requires `gopls` installed (`go install golang.org/x/tools/gopls@latest`). Without gopls, Go falls back to Basic semantic support.
```

**Step 4: Verify docs render correctly**

Visual inspection of the markdown.

**Step 5: Commit**

```bash
git add docs/TLDR.md
git commit -m "docs: update Go to Full semantic support

- Change Go from ⚠️ Basic to ✅ Full in language table
- Add note about gopls requirement

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Run Full Test Suite

**Files:** None (verification only)

**Step 1: Run all tests**

```bash
source .venv/bin/activate && pytest tests/ -v --tb=short
```

**Step 2: Verify all new tests pass**

Expected: All gopls tests pass (or skip if no gopls), no regressions.

**Step 3: Run Go-specific tests**

```bash
source .venv/bin/activate && pytest tests/test_gopls_client.py tests/test_go_semantic.py tests/test_go_call_graph.py -v
```

**Step 4: Commit any fixes if needed**

**Step 5: Final commit message**

```bash
git log --oneline -10
# Should show all the commits from this implementation
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | GoplsClient core | gopls_client.py |
| 2 | Start/stop lifecycle | gopls_client.py |
| 3 | Hover (signatures) | gopls_client.py |
| 4 | Call hierarchy | gopls_client.py |
| 5 | Client management | gopls_client.py |
| 6 | semantic.py signatures | semantic.py |
| 7 | semantic.py docstrings | semantic.py |
| 8 | cross_file_calls.py | cross_file_calls.py |
| 9 | Documentation | TLDR.md |
| 10 | Final verification | - |

**Total estimated tasks:** 10
**New files:** 3 (gopls_client.py, test_gopls_client.py, test_go_semantic.py, test_go_call_graph.py)
**Modified files:** 3 (semantic.py, cross_file_calls.py, TLDR.md)
