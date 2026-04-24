"""Tests for the atomic materializer module."""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SILICONFLOW_API_KEY", "test-key")

from src.materializer import Materializer
from src.registry import KnowledgeRegistry


@pytest.fixture
def tmp_env(tmp_path: Path, monkeypatch):
    """Set up temp directories and registry."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    compiled_dir = tmp_path / "compiled_library"
    db_path = tmp_path / "test.db"

    monkeypatch.setattr("src.config.COMPILED_DIR", compiled_dir)
    monkeypatch.setattr("src.config.DATA_DIR", data_dir)

    registry = KnowledgeRegistry(db_path)
    yield compiled_dir, registry
    registry.close()


def _setup_docs_and_clusters(
    registry: KnowledgeRegistry,
) -> tuple[int, dict[str, str], dict[str, str]]:
    """Create sample docs and clusters, return (compile_run_id, short_doc_texts, doc_summaries)."""
    # Short doc
    sid = registry.register_document("data/it.md", "md", "hash1", title="IT设备申请")
    registry.update_document_status(sid, "embedded", tier="short")

    # Long doc
    lid = registry.register_document("data/travel.md", "md", "hash2", title="差旅报销政策")
    registry.update_document_status(lid, "embedded", tier="long", pageindex_id="uuid-abc-123")

    run_id = registry.create_compile_run("manual")
    registry.save_clusters(
        [
            {"name": "it-hr", "description": "IT和HR政策", "doc_ids": [sid]},
            {"name": "travel", "description": "差旅政策", "doc_ids": [lid]},
        ],
        run_id,
    )
    registry.complete_compile_run(run_id)
    short_doc_texts = {sid: "IT设备申请的详细内容..."}
    doc_summaries = {
        sid: "IT设备申请流程 | 所有员工如需申请新的办公设备",
        lid: "第一章 总则、第二章 交通费用 | 为规范公司员工差旅报销行为",
    }
    return run_id, short_doc_texts, doc_summaries


def test_materialize_creates_skill_md(tmp_env):
    """SKILL.md created with cluster listing and doc titles."""
    compiled_dir, registry = tmp_env
    run_id, short_texts, doc_summaries = _setup_docs_and_clusters(registry)

    mat = Materializer()
    mat.materialize(run_id, registry, short_texts, doc_summaries)

    skill_md = compiled_dir / "SKILL.md"
    assert skill_md.exists()
    content = skill_md.read_text(encoding="utf-8")
    assert "it-hr" in content
    assert "travel" in content
    assert "包含文档" in content


def test_materialize_creates_index_md(tmp_env):
    """INDEX.md per cluster with summaries for fast relevance checking."""
    compiled_dir, registry = tmp_env
    run_id, short_texts, doc_summaries = _setup_docs_and_clusters(registry)

    mat = Materializer()
    mat.materialize(run_id, registry, short_texts, doc_summaries)

    it_index = compiled_dir / "it-hr" / "INDEX.md"
    assert it_index.exists()
    content = it_index.read_text(encoding="utf-8")
    assert "## 短文档" in content
    assert "来源" in content
    assert ">" in content  # summary line present

    travel_index = compiled_dir / "travel" / "INDEX.md"
    assert travel_index.exists()
    content = travel_index.read_text(encoding="utf-8")
    assert "## 长文档" in content
    assert "uuid-abc-123" in content
    assert ">" in content  # summary line present


def test_materialize_creates_short_docs_db(tmp_env):
    """short_docs_db.json created for short docs with content."""
    compiled_dir, registry = tmp_env
    run_id, short_texts, doc_summaries = _setup_docs_and_clusters(registry)

    mat = Materializer()
    mat.materialize(run_id, registry, short_texts, doc_summaries)

    db_file = compiled_dir / "short_docs_db.json"
    assert db_file.exists()
    data = json.loads(db_file.read_text(encoding="utf-8"))
    assert len(data) >= 1
    # Verify content field exists
    entry = list(data.values())[0]
    assert "content" in entry
    assert entry["content"] == "IT设备申请的详细内容..."


def test_materialize_atomic_swap(tmp_env):
    """Atomic swap: old compiled_library/ moved to .bak."""
    compiled_dir, registry = tmp_env
    run_id, short_texts, doc_summaries = _setup_docs_and_clusters(registry)

    # Create a fake existing compiled_library with a marker file
    compiled_dir.mkdir(parents=True, exist_ok=True)
    (compiled_dir / "old_file.txt").write_text("old", encoding="utf-8")

    mat = Materializer()
    mat.materialize(run_id, registry, short_texts, doc_summaries)

    bak_dir = compiled_dir.parent / "compiled_library.bak"
    assert bak_dir.exists()
    assert (bak_dir / "old_file.txt").exists()
    # New files exist
    assert (compiled_dir / "SKILL.md").exists()
