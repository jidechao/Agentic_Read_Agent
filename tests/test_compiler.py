"""Tests for the incremental compilation engine.

Only tests scan/change-detection logic -- no real API calls.
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.compiler import FileInfo, KnowledgeCompiler, ScanResult
from src.registry import KnowledgeRegistry


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_env(tmp_path):
    """Create temp data dir + temp registry."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = tmp_path / "test.db"
    registry = KnowledgeRegistry(db_path)
    compiler = KnowledgeCompiler(registry=registry)
    yield compiler, data_dir, registry
    registry.close()


def _create_file(directory, name, content="test content"):
    f = directory / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content, encoding="utf-8")
    return f


# ── Scan / change-detection tests ────────────────────────────────────────


def test_scan_detects_new_files(tmp_env):
    """Empty registry + files -> all in to_process."""
    compiler, data_dir, _ = tmp_env
    _create_file(data_dir, "doc1.md", "# Title\nContent")
    _create_file(data_dir, "doc2.pdf", "fake pdf")

    changes = compiler.scan_data_dir(data_dir=data_dir)
    assert len(changes.to_process) == 2
    assert len(changes.skipped) == 0
    assert len(changes.deleted) == 0


def test_force_compile_preserves_categories_until_rebuild(tmp_env, monkeypatch):
    """Force compile must let _compile_clusters snapshot old category names first."""
    compiler, data_dir, registry = tmp_env
    _create_file(data_dir, "doc.md", "# Hello")
    registry.upsert_category(
        "annual-health-check",
        "年度体检",
        "旧描述",
        np.array([1.0, 0.0], dtype=np.float32),
        doc_count=1,
    )

    monkeypatch.setattr(compiler, "process_document", lambda _fi: None)

    def fake_compile_clusters(run_id):
        assert run_id > 0
        assert [c["canonical_name"] for c in registry.list_categories()] == [
            "annual-health-check"
        ]
        return 1

    monkeypatch.setattr(compiler, "_compile_clusters", fake_compile_clusters)

    result = compiler.compile(force=True, data_dir=data_dir)

    assert result.clusters_created == 1


def test_scan_returns_scan_result(tmp_env):
    """scan_data_dir returns a ScanResult dataclass, not a plain dict."""
    compiler, data_dir, _ = tmp_env
    changes = compiler.scan_data_dir(data_dir=data_dir)
    assert isinstance(changes, ScanResult)


def test_scan_ignores_non_compiled_deleted_docs(tmp_env):
    """Docs in error/ingested status are NOT marked as deleted even if absent from disk."""
    compiler, data_dir, registry = tmp_env
    id_error = registry.register_document("error.md", "md", "err_hash")
    registry.update_document_status(id_error, "error")
    id_ingested = registry.register_document("ingested.md", "md", "ing_hash")
    # ingested status is the default, no update needed

    changes = compiler.scan_data_dir(data_dir=data_dir)
    assert id_error not in changes.deleted
    assert id_ingested not in changes.deleted


def test_scan_skips_unchanged(tmp_env):
    """Registry has compiled doc with matching hash -> skipped."""
    compiler, data_dir, registry = tmp_env
    f = _create_file(data_dir, "doc.md", "# Hello")
    content_hash = hashlib.sha256(f.read_bytes()).hexdigest()
    doc_id = registry.register_document("doc.md", "md", content_hash)
    registry.cache_embedding(doc_id, np.zeros(1536, dtype=np.float32))
    registry.update_document_status(doc_id, "compiled")

    changes = compiler.scan_data_dir(data_dir=data_dir)
    assert len(changes.skipped) == 1
    assert len(changes.to_process) == 0


def test_scan_detects_changed_file(tmp_env):
    """Registry has doc but hash changed -> to_process."""
    compiler, data_dir, registry = tmp_env
    _create_file(data_dir, "doc.md", "# New content")
    old_hash = hashlib.sha256(b"old content").hexdigest()
    doc_id = registry.register_document("doc.md", "md", old_hash)
    registry.update_document_status(doc_id, "compiled")

    changes = compiler.scan_data_dir(data_dir=data_dir)
    assert len(changes.to_process) == 1
    assert len(changes.skipped) == 0


def test_scan_detects_deleted_file(tmp_env):
    """Registry has doc but file gone -> deleted."""
    compiler, data_dir, registry = tmp_env
    doc_id = registry.register_document("gone.md", "md", "some_hash")
    registry.update_document_status(doc_id, "compiled")

    changes = compiler.scan_data_dir(data_dir=data_dir)
    assert len(changes.deleted) == 1
    assert changes.deleted[0] == doc_id


def test_scan_force_processes_all(tmp_env):
    """force=True -> all files in to_process even if unchanged."""
    compiler, data_dir, registry = tmp_env
    f = _create_file(data_dir, "doc.md", "# Hello")
    content_hash = hashlib.sha256(f.read_bytes()).hexdigest()
    doc_id = registry.register_document("doc.md", "md", content_hash)
    registry.update_document_status(doc_id, "compiled")

    changes = compiler.scan_data_dir(force=True, data_dir=data_dir)
    assert len(changes.to_process) == 1


def test_scan_ignores_unsupported_formats(tmp_env):
    """Unsupported file extensions are ignored."""
    compiler, data_dir, _ = tmp_env
    _create_file(data_dir, "doc.md", "# Hello")
    _create_file(data_dir, "image.png", "fake image data")
    _create_file(data_dir, "data.csv", "a,b,c")

    changes = compiler.scan_data_dir(data_dir=data_dir)
    assert len(changes.to_process) == 1  # only .md


def test_scan_nested_directories(tmp_env):
    """Scan recurses into subdirectories."""
    compiler, data_dir, _ = tmp_env
    _create_file(data_dir / "sub", "a.md", "# A")
    _create_file(data_dir / "sub" / "deep", "b.html", "<h1>B</h1>")

    changes = compiler.scan_data_dir(data_dir=data_dir)
    assert len(changes.to_process) == 2
    paths = {fi.relative_path for fi in changes.to_process}
    assert str(Path("sub") / "a.md") in paths
    assert str(Path("sub") / "deep" / "b.html") in paths


def test_scan_mixed_scenarios(tmp_env):
    """New file + unchanged file + changed file + deleted file all at once."""
    compiler, data_dir, registry = tmp_env

    # Unchanged: registered with matching hash and compiled status
    f_unchanged = _create_file(data_dir, "unchanged.md", "same content")
    h_unchanged = hashlib.sha256(f_unchanged.read_bytes()).hexdigest()
    doc_unchanged = registry.register_document("unchanged.md", "md", h_unchanged)
    registry.cache_embedding(doc_unchanged, np.zeros(1536, dtype=np.float32))
    registry.update_document_status(doc_unchanged, "compiled")

    # Changed: registered with old hash, file now has new content
    f_changed = _create_file(data_dir, "changed.md", "new content")
    old_hash = hashlib.sha256(b"old content").hexdigest()
    registry.register_document("changed.md", "md", old_hash)
    registry.update_document_status(
        registry.find_by_hash(old_hash)["id"], "compiled"
    )

    # New: file present, not in registry
    _create_file(data_dir, "new.html", "<h1>New</h1>")

    # Deleted: in registry, no file on disk
    doc_deleted = registry.register_document("deleted.docx", "docx", "del_hash")
    registry.update_document_status(doc_deleted, "compiled")

    changes = compiler.scan_data_dir(data_dir=data_dir)

    assert len(changes.skipped) == 1
    assert len(changes.to_process) == 2  # changed + new
    # Both the truly deleted doc AND the changed doc (old hash missing) are
    # detected as deleted since their hashes no longer appear on disk.
    assert len(changes.deleted) == 2
    assert doc_deleted in changes.deleted


def test_fileinfo_dataclass_fields(tmp_env):
    """FileInfo contains path, content_hash, and relative_path."""
    compiler, data_dir, _ = tmp_env
    f = _create_file(data_dir, "doc.md", "content")
    expected_hash = hashlib.sha256(f.read_bytes()).hexdigest()

    changes = compiler.scan_data_dir(data_dir=data_dir)
    fi = changes.to_process[0]

    assert isinstance(fi, FileInfo)
    assert fi.path == f
    assert fi.content_hash == expected_hash
    assert fi.relative_path == "doc.md"


def test_scan_empty_data_dir(tmp_env):
    """Empty data dir with no files produces empty results."""
    compiler, data_dir, _ = tmp_env

    changes = compiler.scan_data_dir(data_dir=data_dir)
    assert changes.to_process == []
    assert changes.skipped == []
    assert changes.deleted == []


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
