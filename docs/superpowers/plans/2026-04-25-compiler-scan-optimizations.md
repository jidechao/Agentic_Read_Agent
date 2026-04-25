# Compiler Scan Optimizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize `scan_data_dir()` for memory (streaming hash), query performance (targeted SQL), and type safety (ScanResult dataclass).

**Architecture:** Three independent changes to compiler.py and registry.py, followed by an atomic test update. Each task is independently verifiable.

**Tech Stack:** Python 3.11+, SQLite, pytest

---

### Task 1: Add `find_deleted_doc_ids` to KnowledgeRegistry

**Files:**
- Modify: `src/registry.py` — add method after `find_by_pageindex_id` (after line 184)
- Modify: `tests/test_registry.py` — add test before compile-runs section (after line 112)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_registry.py` after the `test_find_by_hash` function (after line 111):

```python
def test_find_deleted_doc_ids(registry: KnowledgeRegistry) -> None:
    """find_deleted_doc_ids returns IDs of compiled docs whose hash is absent."""
    id_alive = registry.register_document(
        source_path="/data/a.txt", format="text", content_hash="h_alive"
    )
    id_dead = registry.register_document(
        source_path="/data/b.txt", format="text", content_hash="h_dead"
    )
    id_error = registry.register_document(
        source_path="/data/c.txt", format="text", content_hash="h_error"
    )
    registry.update_document_status(id_alive, "compiled")
    registry.update_document_status(id_dead, "compiled")
    registry.update_document_status(id_error, "error")

    # Only id_dead's hash is missing from seen_hashes
    result = registry.find_deleted_doc_ids({"h_alive", "h_error"})
    assert id_dead in result
    assert id_alive not in result
    assert id_error not in result  # error status excluded


def test_find_deleted_doc_ids_empty_seen(registry: KnowledgeRegistry) -> None:
    """Empty seen_hashes returns all compiled doc IDs."""
    id_a = registry.register_document(
        source_path="/data/a.txt", format="text", content_hash="ha"
    )
    registry.update_document_status(id_a, "compiled")
    result = registry.find_deleted_doc_ids(set())
    assert id_a in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_registry.py::test_find_deleted_doc_ids -v`
Expected: FAIL with `AttributeError: 'KnowledgeRegistry' object has no attribute 'find_deleted_doc_ids'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/registry.py` after `find_by_pageindex_id` (after line 184):

```python
def find_deleted_doc_ids(self, seen_hashes: set[str]) -> list[str]:
    """Return IDs of compiled docs whose content_hash is not in seen_hashes.

    If seen_hashes is empty (empty data dir), all compiled doc IDs are returned.
    """
    conn = self._get_conn()
    if not seen_hashes:
        rows = conn.execute(
            "SELECT id FROM documents WHERE status = 'compiled'"
        ).fetchall()
    else:
        placeholders = ",".join("?" for _ in seen_hashes)
        rows = conn.execute(
            f"SELECT id FROM documents WHERE status = 'compiled' "
            f"AND content_hash NOT IN ({placeholders})",
            list(seen_hashes),
        ).fetchall()
    return [row["id"] for row in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_registry.py -v`
Expected: All tests PASS (including the two new ones)

- [ ] **Step 5: Commit**

```bash
git add src/registry.py tests/test_registry.py
git commit -m "feat: add find_deleted_doc_ids to KnowledgeRegistry"
```

---

### Task 2: Add streaming `_sha256_file` helper

**Files:**
- Modify: `src/compiler.py` — add function in the Helpers section (after line 27)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_compiler.py` at the end of the file:

```python
def test_streaming_hash_matches_standard(tmp_path):
    """_sha256_file produces the same result as hashlib for various sizes."""
    from src.compiler import _sha256_file

    # Small file (< 1 chunk)
    small = tmp_path / "small.bin"
    small.write_bytes(b"hello world")
    assert _sha256_file(small) == hashlib.sha256(b"hello world").hexdigest()

    # Multi-chunk file (> 64KB)
    big = tmp_path / "big.bin"
    data = b"x" * 200_000
    big.write_bytes(data)
    assert _sha256_file(big) == hashlib.sha256(data).hexdigest()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_compiler.py::test_streaming_hash_matches_standard -v`
Expected: FAIL with `ImportError: cannot import name '_sha256_file'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/compiler.py` in the Helpers section (after line 27, before the `_find_heading` function):

```python
def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash using constant 64KB memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
```

Then update line 157 to use it. Change:

```python
            content_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
```

to:

```python
            content_hash = _sha256_file(fpath)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_compiler.py -v`
Expected: All tests PASS (including the new `test_streaming_hash_matches_standard`)

- [ ] **Step 5: Commit**

```bash
git add src/compiler.py tests/test_compiler.py
git commit -m "feat: add streaming _sha256_file helper for constant-memory hashing"
```

---

### Task 3: Add `ScanResult` dataclass and switch all call sites

This is the atomic switch — `ScanResult` + `scan_data_dir` return type + `compile()` + all tests change together.

**Files:**
- Modify: `src/compiler.py` — add `ScanResult` dataclass, update `scan_data_dir`, update `compile()`
- Modify: `tests/test_compiler.py` — switch all `changes["key"]` to `changes.key`

- [ ] **Step 1: Add `ScanResult` dataclass**

In `src/compiler.py`, add after the `FileInfo` dataclass (after line 61):

```python
@dataclass
class ScanResult:
    """Typed result of a data directory scan."""

    to_process: list[FileInfo]
    skipped: list[FileInfo]
    deleted: list[str]
```

- [ ] **Step 2: Update `scan_data_dir` return type and body**

In `src/compiler.py`, change the `scan_data_dir` signature (line 142):

From:
```python
    ) -> dict[str, list]:
```

To:
```python
    ) -> ScanResult:
```

Update the docstring to mention `ScanResult`.

Update the deleted-detection block (lines 172-178). Change:

```python
        # Detect deleted: docs in registry not found in current scan
        all_docs = self.registry.list_documents()
        deleted: list[str] = []
        for doc in all_docs:
            if doc["content_hash"] not in seen_hashes:
                deleted.append(doc["id"])

        return {"to_process": to_process, "skipped": skipped, "deleted": deleted}
```

To:
```python
        # Detect deleted: compiled docs whose hash is absent from current scan
        deleted = self.registry.find_deleted_doc_ids(seen_hashes)

        return ScanResult(to_process=to_process, skipped=skipped, deleted=deleted)
```

- [ ] **Step 3: Update `compile()` to use `ScanResult` attributes**

In `src/compiler.py`, update the `compile()` method. Change all `changes["key"]` to `changes.key`:

Line 103: `for fi in changes["to_process"]:` → `for fi in changes.to_process:`
Line 106: `for doc_id in changes["deleted"]:` → `for doc_id in changes.deleted:`
Line 110: `if changes["to_process"] or changes["deleted"]:` → `if changes.to_process or changes.deleted:`
Line 124: `docs_processed=len(changes["to_process"]),` → `docs_processed=len(changes.to_process),`
Line 125: `docs_skipped=len(changes["skipped"]),` → `docs_skipped=len(changes.skipped),`
Line 129: `docs_processed=len(changes["to_process"]),` → `docs_processed=len(changes.to_process),`
Line 130: `docs_skipped=len(changes["skipped"]),` → `docs_skipped=len(changes.skipped),`
Line 131: `docs_deleted=len(changes["deleted"]),` → `docs_deleted=len(changes.deleted),`

- [ ] **Step 4: Update `tests/test_compiler.py` to use attribute access**

Switch all `changes["key"]` to `changes.key` throughout the file. These are the lines:

| Line | Old | New |
|------|-----|-----|
| 47 | `changes["to_process"]` | `changes.to_process` |
| 48 | `changes["skipped"]` | `changes.skipped` |
| 49 | `changes["deleted"]` | `changes.deleted` |
| 62 | `changes["skipped"]` | `changes.skipped` |
| 63 | `changes["to_process"]` | `changes.to_process` |
| 75 | `changes["to_process"]` | `changes.to_process` |
| 76 | `changes["skipped"]` | `changes.skipped` |
| 86 | `changes["deleted"]` | `changes.deleted` |
| 87 | `changes["deleted"]` | `changes.deleted` |
| 99 | `changes["to_process"]` | `changes.to_process` |
| 110 | `changes["to_process"]` | `changes.to_process` |
| 120 | `changes["to_process"]` | `changes.to_process` |
| 121 | `changes["to_process"]` | `changes.to_process` |
| 154 | `changes["skipped"]` | `changes.skipped` |
| 155 | `changes["to_process"]` | `changes.to_process` |
| 158 | `changes["deleted"]` | `changes.deleted` |
| 159 | `changes["deleted"]` | `changes.deleted` |
| 169 | `changes["to_process"]` | `changes.to_process` |
| 182 | `changes["to_process"]` | `changes.to_process` |
| 183 | `changes["skipped"]` | `changes.skipped` |
| 184 | `changes["deleted"]` | `changes.deleted` |

Also update the import on line 11 to include `ScanResult`:

```python
from src.compiler import FileInfo, KnowledgeCompiler, ScanResult
```

And add a type assertion at the end of `test_scan_detects_new_files` (or as a new test):

```python
def test_scan_returns_scan_result(tmp_env):
    """scan_data_dir returns a ScanResult dataclass, not a plain dict."""
    compiler, data_dir, _ = tmp_env
    changes = compiler.scan_data_dir(data_dir=data_dir)
    assert isinstance(changes, ScanResult)
```

Add this test after `test_scan_detects_new_files`.

- [ ] **Step 5: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/compiler.py tests/test_compiler.py
git commit -m "refactor: add ScanResult dataclass and switch scan_data_dir to typed returns"
```
