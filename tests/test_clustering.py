"""Tests for the hybrid clustering pipeline.

These tests avoid real API calls. They lock down the local behavior that makes
the LLM-assisted pipeline stable: normalized cosine geometry, persistent
category identity, incremental assignment, and report materialization.
"""
from pathlib import Path

import numpy as np
import pytest

from src.clustering.coarse import cosine_agglomerative, l2_normalize
from src.clustering.discovery import DiscoveredCategory
from src.clustering.evaluator import build_cluster_report
from src.clustering.llm_assign import assign_to_existing_category
from src.clustering.review import merge_singleton_categories_by_keywords
from src.compiler import KnowledgeCompiler
from src.materializer import Materializer
from src.registry import KnowledgeRegistry


@pytest.fixture
def registry(tmp_path: Path) -> KnowledgeRegistry:
    reg = KnowledgeRegistry(tmp_path / "knowledge.db")
    yield reg
    reg.close()


def test_registry_persists_stable_categories(registry: KnowledgeRegistry) -> None:
    """Categories have stable ids and document assignments across reads."""
    doc_id = registry.register_document("travel.md", "md", "hash-travel", title="差旅政策")
    registry.update_document_status(doc_id, "embedded", tier="long")

    centroid = l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    category_id = registry.upsert_category(
        canonical_name="travel-policy",
        display_name="差旅政策",
        description="差旅、住宿与报销政策",
        centroid=centroid,
        doc_count=1,
    )
    registry.assign_document_category(
        doc_id,
        category_id,
        similarity=0.93,
        assigned_by="embedding",
        confidence=0.93,
    )

    categories = registry.list_categories()
    assert [c["id"] for c in categories] == [category_id]
    assert categories[0]["canonical_name"] == "travel-policy"
    assert np.allclose(registry.get_category_centroid(category_id), centroid)

    docs = registry.list_category_documents(category_id)
    assert [d["id"] for d in docs] == [doc_id]
    assert docs[0]["similarity"] == pytest.approx(0.93)


def test_topic_summary_cache_is_keyed_by_content_hash(
    registry: KnowledgeRegistry,
) -> None:
    registry.cache_topic_summary("hash-a", "这是一份关于 VPN 远程接入的 IT 指南")
    assert registry.get_topic_summary("hash-a") == "这是一份关于 VPN 远程接入的 IT 指南"
    assert registry.get_topic_summary("hash-missing") is None


def test_cosine_agglomerative_separates_unrelated_pdf_topics() -> None:
    """Two semantically distant PDF groups should not be forced into one cluster."""
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],   # PageIndex README
            [0.98, 0.02, 0.0], # another PageIndex doc
            [0.0, 1.0, 0.0],   # national standard
            [0.0, 0.97, 0.03], # another accounting standard
        ],
        dtype=np.float32,
    )
    labels = cosine_agglomerative(matrix, threshold=0.2)

    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_assign_to_existing_category_uses_cosine_threshold(
    registry: KnowledgeRegistry,
) -> None:
    travel_id = registry.upsert_category(
        canonical_name="travel-policy",
        display_name="差旅政策",
        description="差旅、住宿与报销政策",
        centroid=l2_normalize(np.array([1.0, 0.0], dtype=np.float32)),
        doc_count=2,
    )
    registry.upsert_category(
        canonical_name="it-operations",
        display_name="IT 运维",
        description="VPN、设备与账号申请",
        centroid=l2_normalize(np.array([0.0, 1.0], dtype=np.float32)),
        doc_count=2,
    )

    result = assign_to_existing_category(
        l2_normalize(np.array([0.91, 0.09], dtype=np.float32)),
        registry.list_categories(),
        threshold=0.55,
    )

    assert result is not None
    assert result.category_id == travel_id
    assert result.similarity > 0.9

    low_confidence = assign_to_existing_category(
        l2_normalize(np.array([0.5, 0.5], dtype=np.float32)),
        registry.list_categories(),
        threshold=0.9,
    )
    assert low_confidence is None


def test_materializer_uses_categories_and_writes_cluster_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry: KnowledgeRegistry,
) -> None:
    compiled_dir = tmp_path / "compiled_library"
    monkeypatch.setattr("src.config.COMPILED_DIR", compiled_dir)

    doc_a = registry.register_document("README_CN.pdf", "pdf", "hash-readme", title="PageIndex")
    doc_b = registry.register_document("demo.pdf", "pdf", "hash-standard", title="中华人民共和国国家标准")
    registry.update_document_status(doc_a, "embedded", tier="long", pageindex_id="pi-readme")
    registry.update_document_status(doc_b, "embedded", tier="long", pageindex_id="pi-standard")
    registry.cache_embedding(doc_a, l2_normalize(np.array([1.0, 0.0], dtype=np.float32)))
    registry.cache_embedding(doc_b, l2_normalize(np.array([0.0, 1.0], dtype=np.float32)))

    readme_cat = registry.upsert_category(
        canonical_name="pageindex-docs",
        display_name="PageIndex 文档",
        description="PageIndex 使用说明与开发文档",
        centroid=l2_normalize(np.array([1.0, 0.0], dtype=np.float32)),
        doc_count=1,
    )
    standard_cat = registry.upsert_category(
        canonical_name="national-standards",
        display_name="国家标准",
        description="国家标准和规范文档",
        centroid=l2_normalize(np.array([0.0, 1.0], dtype=np.float32)),
        doc_count=1,
    )
    registry.assign_document_category(doc_a, readme_cat, 0.99, "llm", 0.95)
    registry.assign_document_category(doc_b, standard_cat, 0.99, "llm", 0.95)

    run_id = registry.create_compile_run("manual")
    Materializer().materialize(
        run_id,
        registry,
        doc_summaries={
            doc_a: "PageIndex 是一个文档索引和检索工具",
            doc_b: "电子凭证会计档案封装技术要求",
        },
    )

    skill = (compiled_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "pageindex-docs" in skill
    assert "national-standards" in skill

    report = (compiled_dir / "CLUSTER_REPORT.md").read_text(encoding="utf-8")
    assert "PageIndex 文档" in report
    assert "国家标准" in report
    assert "README_CN.pdf" in report
    assert "demo.pdf" in report


def test_cluster_report_flags_low_similarity_docs(
    registry: KnowledgeRegistry,
) -> None:
    doc_id = registry.register_document("mixed.md", "md", "hash-mixed", title="混合主题")
    registry.update_document_status(doc_id, "embedded", tier="short")
    registry.cache_embedding(doc_id, l2_normalize(np.array([0.0, 1.0], dtype=np.float32)))
    category_id = registry.upsert_category(
        canonical_name="travel-policy",
        display_name="差旅政策",
        description="差旅、住宿与报销政策",
        centroid=l2_normalize(np.array([1.0, 0.0], dtype=np.float32)),
        doc_count=1,
    )
    registry.assign_document_category(doc_id, category_id, 0.2, "embedding", 0.2)

    report = build_cluster_report(registry)
    assert "可疑文档" in report
    assert "mixed.md" in report


def test_merge_singleton_categories_by_keywords_groups_related_docs() -> None:
    categories = [
        {"id": "c1", "canonical_name": "travel", "doc_count": 1},
        {"id": "c2", "canonical_name": "hotel", "doc_count": 1},
        {"id": "c3", "canonical_name": "vpn", "doc_count": 1},
        {"id": "c4", "canonical_name": "equipment", "doc_count": 1},
        {"id": "c5", "canonical_name": "standard", "doc_count": 1},
    ]
    category_docs = {
        "c1": [{"title": "2026年公司差旅报销政策", "source_path": "travel.md"}],
        "c2": [{"title": "出差住宿标准", "source_path": "hotel.md"}],
        "c3": [{"title": "VPN 远程接入指南", "source_path": "vpn.md"}],
        "c4": [{"title": "IT 设备申请流程", "source_path": "it.md"}],
        "c5": [{"title": "中华人民共和国国家标准", "source_path": "gb.pdf"}],
    }

    groups = merge_singleton_categories_by_keywords(categories, category_docs)

    merged_sets = {frozenset(group) for group in groups}
    assert frozenset({"c1", "c2"}) in merged_sets
    assert frozenset({"c3", "c4"}) in merged_sets
    assert frozenset({"c5"}) in merged_sets


def test_full_compile_single_document_creates_category(
    registry: KnowledgeRegistry,
) -> None:
    doc_id = registry.register_document("solo.md", "md", "hash-solo", title="单篇制度")
    registry.update_document_status(doc_id, "embedded", tier="short")
    registry.cache_embedding(doc_id, l2_normalize(np.array([1.0, 0.0], dtype=np.float32)))

    compiler = KnowledgeCompiler(registry=registry)
    run_id = registry.create_compile_run("manual")

    assert compiler._compile_clusters(run_id) == 1

    categories = registry.list_categories()
    assert len(categories) == 1
    docs = registry.list_category_documents(categories[0]["id"])
    assert [doc["id"] for doc in docs] == [doc_id]


def test_full_compile_zero_documents_clears_stale_categories(
    registry: KnowledgeRegistry,
) -> None:
    stale_id = registry.upsert_category(
        canonical_name="stale",
        display_name="旧分类",
        description="旧数据",
        centroid=l2_normalize(np.array([1.0, 0.0], dtype=np.float32)),
        doc_count=1,
    )
    assert stale_id

    compiler = KnowledgeCompiler(registry=registry)
    run_id = registry.create_compile_run("manual")

    assert compiler._compile_clusters(run_id) == 0
    assert registry.list_categories() == []


def test_incremental_materialize_preserves_existing_short_doc_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry: KnowledgeRegistry,
) -> None:
    compiled_dir = tmp_path / "compiled_library"
    monkeypatch.setattr("src.config.COMPILED_DIR", compiled_dir)

    old_doc = registry.register_document("old.md", "md", "hash-old", title="旧短文档")
    new_doc = registry.register_document("new.md", "md", "hash-new", title="新短文档")
    registry.update_document_status(old_doc, "compiled", tier="short")
    registry.update_document_status(new_doc, "embedded", tier="short")
    old_cat = registry.upsert_category(
        "old-category",
        "旧分类",
        "旧短文档分类",
        l2_normalize(np.array([1.0, 0.0], dtype=np.float32)),
        doc_count=1,
    )
    new_cat = registry.upsert_category(
        "new-category",
        "新分类",
        "新短文档分类",
        l2_normalize(np.array([0.0, 1.0], dtype=np.float32)),
        doc_count=1,
    )
    registry.assign_document_category(old_doc, old_cat, 1.0, "llm", 1.0)
    registry.assign_document_category(new_doc, new_cat, 1.0, "llm", 1.0)

    run_id = registry.create_compile_run("manual")
    Materializer().materialize(run_id, registry, {old_doc: "旧短文档全文"})

    second_run = registry.create_compile_run("manual")
    Materializer().materialize(second_run, registry, {new_doc: "新短文档全文"})

    data = __import__("json").loads(
        (compiled_dir / "short_docs_db.json").read_text(encoding="utf-8")
    )
    assert data[old_doc]["content"] == "旧短文档全文"
    assert data[new_doc]["content"] == "新短文档全文"


def test_full_rebuild_preserves_category_id_for_same_canonical_name(
    registry: KnowledgeRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = registry.register_document("travel.md", "md", "hash-travel", title="差旅政策")
    doc_b = registry.register_document("hotel.md", "md", "hash-hotel", title="住宿标准")
    registry.update_document_status(doc_id, "embedded", tier="short")
    registry.update_document_status(doc_b, "embedded", tier="short")
    vector = l2_normalize(np.array([1.0, 0.0], dtype=np.float32))
    vector_b = l2_normalize(np.array([0.9, 0.1], dtype=np.float32))
    registry.cache_embedding(doc_id, vector)
    registry.cache_embedding(doc_b, vector_b)
    existing_id = registry.upsert_category(
        "travel-policy",
        "差旅政策",
        "旧描述",
        vector,
        doc_count=1,
    )
    registry.assign_document_category(doc_id, existing_id, 1.0, "llm", 1.0)

    def fake_discover(*_args, **_kwargs):
        return [
            DiscoveredCategory(
                canonical_name="travel-policy",
                display_name="差旅政策",
                description="新描述",
                doc_ids=[doc_id, doc_b],
                centroid=l2_normalize(np.array([0.95, 0.05], dtype=np.float32)),
            )
        ]

    monkeypatch.setattr("src.compiler.discover_categories", fake_discover)
    compiler = KnowledgeCompiler(registry=registry)
    run_id = registry.create_compile_run("manual")

    compiler._compile_clusters(run_id)

    categories = registry.list_categories()
    assert [category["id"] for category in categories] == [existing_id]
    assert categories[0]["description"] == "新描述"


def test_full_rebuild_merges_duplicate_canonical_names(
    registry: KnowledgeRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_a = registry.register_document("a.md", "md", "hash-a", title="制度 A")
    doc_b = registry.register_document("b.md", "md", "hash-b", title="制度 B")
    vec_a = l2_normalize(np.array([1.0, 0.0], dtype=np.float32))
    vec_b = l2_normalize(np.array([0.9, 0.1], dtype=np.float32))
    for doc_id, vec in ((doc_a, vec_a), (doc_b, vec_b)):
        registry.update_document_status(doc_id, "embedded", tier="short")
        registry.cache_embedding(doc_id, vec)

    def fake_discover(*_args, **_kwargs):
        return [
            DiscoveredCategory("policy", "制度", "第一组", [doc_a], vec_a),
            DiscoveredCategory("policy", "制度", "第二组", [doc_b], vec_b),
        ]

    monkeypatch.setattr("src.compiler.discover_categories", fake_discover)
    compiler = KnowledgeCompiler(registry=registry)
    run_id = registry.create_compile_run("manual")

    compiler._compile_clusters(run_id)

    categories = registry.list_categories()
    assert len(categories) == 1
    docs = registry.list_category_documents(categories[0]["id"])
    assert {doc["id"] for doc in docs} == {doc_a, doc_b}
    assert categories[0]["doc_count"] == 2
