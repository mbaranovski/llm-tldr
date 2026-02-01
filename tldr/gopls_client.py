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
