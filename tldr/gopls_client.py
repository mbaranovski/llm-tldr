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
