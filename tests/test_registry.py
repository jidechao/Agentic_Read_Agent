"""Tests for the SQLite-backed KnowledgeRegistry."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SILICONFLOW_API_KEY", "test-key-for-registry-test")

from src.registry import KnowledgeRegistry


@pytest.fixture
def registry(tmp_path: Path) -> KnowledgeRegistry:
    """Create a KnowledgeRegistry backed by a temporary database."""
    db_path = tmp_path / "test_knowledge.db"
    reg = KnowledgeRegistry(db_path)
    yield reg
    reg.close()


# ── 1. Schema ──────────────────────────────────────────────────────────
def test_create_tables(registry: KnowledgeRegistry) -> None:
    """All four required tables must exist after initialisation."""
    conn = registry._get_conn()
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {row["name"] for row in cursor}
    assert {"documents", "compile_runs", "clusters", "document_clusters"}.issubset(
        tables
    )


# ── 2. Document CRUD ──────────────────────────────────────────────────
def test_register_document(registry: KnowledgeRegistry) -> None:
    doc_id = registry.register_document(
        source_path="/data/readme.md",
        format="markdown",
        content_hash="abc123",
        title="README",
    )
    assert isinstance(doc_id, str) and len(doc_id) > 0

    doc = registry.get_document(doc_id)
    assert doc is not None
    assert doc["id"] == doc_id
    assert doc["source_path"] == "/data/readme.md"
    assert doc["format"] == "markdown"
    assert doc["content_hash"] == "abc123"
    assert doc["title"] == "README"
    assert doc["status"] == "ingested"
    assert doc["tier"] == "unclassified"


def test_update_document_status(registry: KnowledgeRegistry) -> None:
    doc_id = registry.register_document(
        source_path="/data/doc.txt",
        format="text",
        content_hash="h1",
    )
    registry.update_document_status(
        doc_id,
        status="classified",
        tier="short",
        token_count=42,
        has_structure=True,
    )
    doc = registry.get_document(doc_id)
    assert doc["status"] == "classified"
    assert doc["tier"] == "short"
    assert doc["token_count"] == 42
    assert doc["has_structure"] is True


def test_set_document_error(registry: KnowledgeRegistry) -> None:
    doc_id = registry.register_document(
        source_path="/data/bad.pdf",
        format="pdf",
        content_hash="h2",
    )
    registry.update_document_status(
        doc_id, status="error", error_message="parse failure"
    )
    doc = registry.get_document(doc_id)
    assert doc["status"] == "error"
    assert doc["error_message"] == "parse failure"


def test_delete_document(registry: KnowledgeRegistry) -> None:
    doc_id = registry.register_document(
        source_path="/data/gone.txt",
        format="text",
        content_hash="h3",
    )
    registry.delete_document(doc_id)
    assert registry.get_document(doc_id) is None


def test_find_by_hash(registry: KnowledgeRegistry) -> None:
    registry.register_document(
        source_path="/data/a.txt", format="text", content_hash="unique_hash"
    )
    found = registry.find_by_hash("unique_hash")
    assert found is not None
    assert found["content_hash"] == "unique_hash"

    missing = registry.find_by_hash("nonexistent")
    assert missing is None


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


def test_list_documents_by_status(registry: KnowledgeRegistry) -> None:
    id_a = registry.register_document(
        source_path="/data/a.txt", format="text", content_hash="ha"
    )
    id_b = registry.register_document(
        source_path="/data/b.txt", format="text", content_hash="hb"
    )
    registry.update_document_status(id_b, status="classified", tier="short")

    ingested = registry.list_documents(status="ingested")
    assert len(ingested) == 1
    assert ingested[0]["id"] == id_a

    classified = registry.list_documents(status="classified")
    assert len(classified) == 1
    assert classified[0]["id"] == id_b


# ── 3. Embedding cache ─────────────────────────────────────────────────
def test_cache_embedding(registry: KnowledgeRegistry) -> None:
    doc_id = registry.register_document(
        source_path="/data/vec.txt", format="text", content_hash="hv"
    )
    original = np.random.randn(128).astype(np.float32)
    registry.cache_embedding(doc_id, original)

    loaded = registry.get_embedding(doc_id)
    assert loaded is not None
    assert np.allclose(original, loaded)


# ── 4. Compile runs ───────────────────────────────────────────────────
def test_create_compile_run(registry: KnowledgeRegistry) -> None:
    run_id = registry.create_compile_run(trigger_type="manual")
    assert isinstance(run_id, int)

    run = registry.get_compile_run(run_id)
    assert run is not None
    assert run["trigger_type"] == "manual"
    assert run["status"] == "running"


def test_complete_compile_run(registry: KnowledgeRegistry) -> None:
    run_id = registry.create_compile_run(trigger_type="auto")
    registry.complete_compile_run(
        run_id, docs_processed=5, docs_skipped=2
    )
    run = registry.get_compile_run(run_id)
    assert run["status"] == "completed"
    assert run["docs_processed"] == 5
    assert run["docs_skipped"] == 2
    assert run["completed_at"] is not None


# ── 5. Clusters ────────────────────────────────────────────────────────
def test_save_clusters(registry: KnowledgeRegistry) -> None:
    id_a = registry.register_document(
        source_path="/data/ca.txt", format="text", content_hash="c1"
    )
    id_b = registry.register_document(
        source_path="/data/cb.txt", format="text", content_hash="c2"
    )
    run_id = registry.create_compile_run(trigger_type="manual")

    clusters = [
        {"name": "it-hr", "description": "IT and HR docs", "doc_ids": [id_a, id_b]},
    ]
    cluster_ids = registry.save_clusters(clusters, run_id)
    assert len(cluster_ids) == 1

    # Verify document_clusters linkage
    conn = registry._get_conn()
    rows = conn.execute(
        "SELECT doc_id FROM document_clusters WHERE cluster_id = ?",
        (cluster_ids[0],),
    ).fetchall()
    linked_ids = {row["doc_id"] for row in rows}
    assert linked_ids == {id_a, id_b}
