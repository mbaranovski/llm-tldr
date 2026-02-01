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
            request_id = self._request_id
            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }

            try:
                content = json.dumps(request)
                message = f"Content-Length: {len(content)}\r\n\r\n{content}"
                self._process.stdin.write(message.encode())
                self._process.stdin.flush()

                # Keep reading until we get the response for our request ID
                # (skip any notifications from gopls)
                for _ in range(50):  # Max iterations to avoid infinite loop
                    response = self._read_response()
                    if not response:
                        return None

                    # Check if this is our response (has matching id)
                    if response.get("id") == request_id:
                        if "error" in response:
                            logger.warning(f"gopls error: {response['error']}")
                            return None
                        return response.get("result")

                    # Otherwise it's a notification, skip it
                    logger.debug(f"Skipping notification: {response.get('method', 'unknown')}")

                logger.warning("Max iterations reached waiting for response")
                return None

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
            kind = contents.get("kind", "plaintext")
        elif isinstance(contents, str):
            value = contents
            kind = "plaintext"
        else:
            return None

        lines = value.strip().split("\n")
        signature = ""
        doc = ""

        if kind == "markdown":
            # Parse markdown format with code blocks
            in_code_block = False
            for l in lines:
                if l.startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    signature = l.strip()
                else:
                    # Skip markdown horizontal rules
                    if l.strip() == "---":
                        continue
                    if doc:
                        doc += "\n"
                    doc += l
        else:
            # Parse plaintext format: first line is signature, rest is doc
            if lines:
                signature = lines[0].strip()
                if len(lines) > 1:
                    doc = "\n".join(lines[1:])

        if not signature:
            return None

        name, receiver, sym_kind = self._parse_go_signature(signature)

        return GoSymbol(
            name=name,
            qualified_name=f"({receiver}).{name}" if receiver else name,
            signature=signature,
            doc=doc.strip(),
            file=file_path,
            line=line,
            kind=sym_kind,
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
