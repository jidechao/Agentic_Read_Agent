# Compiler Scan Optimizations Design

Optimize `scan_data_dir()` in `src/compiler.py` for memory efficiency, query performance, and type safety.

## Changes

### 1. Streaming SHA-256

Replace `fpath.read_bytes()` with a chunked hash helper.

```python
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
```

Memory usage: O(64KB) constant instead of O(file_size).
Call site: `compiler.py:157` changes from `hashlib.sha256(fpath.read_bytes()).hexdigest()` to `_sha256_file(fpath)`.

### 2. Typed ScanResult

Add `@dataclass` to replace untyped `dict[str, list]` return.

```python
@dataclass
class ScanResult:
    to_process: list[FileInfo]
    skipped: list[FileInfo]
    deleted: list[str]
```

`scan_data_dir` returns `ScanResult` instead of `dict[str, list]`.
`compile()` and all test assertions switch from `changes["to_process"]` to `changes.to_process`.

### 3. Targeted Deleted-Detection Query

Add `find_deleted_doc_ids(seen_hashes: set[str]) -> list[str]` to `KnowledgeRegistry`.

```sql
SELECT id FROM documents WHERE status = 'compiled' AND content_hash NOT IN (?, ?, ...)
```

`compiler.py:173-177` changes from `list_documents()` + Python filter to `self.registry.find_deleted_doc_ids(seen_hashes)`.

Only docs with `status = 'compiled'` are candidates for deletion detection. Docs in `error` or intermediate states are excluded.

Edge case: if `seen_hashes` is empty (empty data dir), return all compiled doc IDs (everything is "deleted").

## Files Touched

| File | Change |
|------|--------|
| `src/compiler.py` | Add `_sha256_file`, add `ScanResult` dataclass, update `scan_data_dir` return type, update `compile()` access pattern |
| `src/registry.py` | Add `find_deleted_doc_ids` method |
| `tests/test_compiler.py` | Switch dict access to attribute access |

## Out of Scope

- No changes to `FileInfo`, `CompileResult`, or other dataclasses
- No ORM or query builder
- No methods on `ScanResult` (pure data carrier)
- No changes to `process_document` or clustering pipeline
