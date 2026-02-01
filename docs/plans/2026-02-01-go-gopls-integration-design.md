# Go Language Full Semantic Support via gopls

**Date:** 2026-02-01
**Status:** Approved
**Goal:** Upgrade Go from "Basic" to "Full" semantic support

## Problem Statement

Go currently has limited semantic analysis:
1. **Signatures are impoverished** - `function processData(...)` instead of full typed signatures
2. **Call graph is broken** for:
   - Same-package cross-file calls (direct calls ignored)
   - Method calls on typed objects (no type resolution)
   - Interface dispatch (no implementation tracking)
3. **No docstring extraction** for Go doc comments

## Solution

Integrate gopls (Go Language Server) to provide type-aware semantic analysis while keeping tree-sitter as a fast fallback.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Go Analysis with gopls                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      GoplsClient (new)                               │    │
│  │  • Subprocess lifecycle management                                   │    │
│  │  • JSON-RPC over stdin/stdout                                        │    │
│  │  • Request batching & caching                                        │    │
│  │  • Graceful fallback if gopls unavailable                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│         ├──────────────────────┬──────────────────────┐                     │
│         ▼                      ▼                      ▼                     │
│  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐               │
│  │  Signatures │       │ Call Graph  │       │  References │               │
│  ├─────────────┤       ├─────────────┤       ├─────────────┤               │
│  │ textDoc/    │       │ callHier/   │       │ textDoc/    │               │
│  │ hover       │       │ incomingCalls│      │ references  │               │
│  │             │       │ outgoingCalls│      │             │               │
│  └─────────────┘       └─────────────┘       └─────────────┘               │
│         │                      │                      │                     │
│         ▼                      ▼                      ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Integration Layer                                 │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  semantic.py        │  cross_file_calls.py  │  hybrid_extractor.py  │    │
│  │  └─ _get_function_  │  └─ _build_go_call_   │  └─ _extract_go_      │    │
│  │     signature()     │     graph()           │     function()        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `tldr/gopls_client.py` | CREATE | LSP client for gopls |
| `tldr/semantic.py` | MODIFY | Use gopls for Go signatures/docstrings |
| `tldr/cross_file_calls.py` | MODIFY | Use gopls call hierarchy |
| `tldr/hybrid_extractor.py` | MODIFY | Enrich extraction with gopls |
| `tests/test_gopls_integration.py` | CREATE | Integration tests |
| `docs/TLDR.md` | MODIFY | Update table (Go → Full) |

## Implementation Phases

### Phase 1: Core Client
- Create `gopls_client.py` with subprocess management
- Implement JSON-RPC communication
- Add hover, callHierarchy methods
- Basic tests for start/stop/hover

### Phase 2: Signature Integration
- Modify `semantic.py` `_get_function_signature()` for Go
- Modify `semantic.py` `_get_function_docstring()` for Go
- Modify `hybrid_extractor.py` to enrich FunctionInfo
- Test: signatures show full types

### Phase 3: Call Graph Integration
- Modify `cross_file_calls.py` `_build_go_call_graph()`
- Use `callHierarchy/outgoingCalls` for accurate call graph
- Test: cross-file calls, method calls, interface dispatch

### Phase 4: Documentation & Cleanup
- Update TLDR.md table
- Add installation docs for gopls
- Final integration tests

## Before/After

### Signatures
```
Before: function processData(...)
After:  func processData(ctx context.Context, data []byte) (Result, error)
```

### Call Graph
```
Before:                          After:
main.go:main                     main.go:main
  └─→ (missing)                    ├─→ server.go:NewServer
                                   ├─→ server.go:(*Server).Start
                                   └─→ handlers.go:RegisterRoutes
```

### Documentation Table
```
| Language | AST | Call Graph | CFG | DFG | PDG | Semantic* |
|----------|-----|------------|-----|-----|-----|-----------|
| Go       | ✅  | ✅         | ✅  | ✅  | ✅  | ✅ Full   |  ← from ⚠️ Basic
```

## Requirements

**Runtime:**
- gopls in PATH: `go install golang.org/x/tools/gopls@latest`
- Go project must have `go.mod`

**Graceful Degradation:**
- No gopls → tree-sitter fallback
- No go.mod → tree-sitter fallback
- gopls error → skip file, continue

## Key Design Decisions

1. **Lazy initialization** - gopls starts only on first Go request
2. **Per-project singleton** - one gopls per project root
3. **Caching** - cache hover/call results
4. **Graceful fallback** - tree-sitter if gopls unavailable

## Test Plan

- `test_gopls_client_lifecycle` - start/stop subprocess
- `test_gopls_hover_signature` - typed signature extraction
- `test_gopls_outgoing_calls` - call hierarchy outgoing
- `test_gopls_incoming_calls` - call hierarchy incoming
- `test_gopls_cross_package_calls` - cross-package resolution
- `test_gopls_method_on_object` - method call resolution
- `test_gopls_interface_implementation` - interface dispatch
- `test_fallback_without_gopls` - graceful degradation
- `test_fallback_without_gomod` - no module fallback
