"""LLM review hooks for suspicious cluster assignments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.clustering.llm_assign import AssignmentResult, assign_to_existing_category


@dataclass(frozen=True)
class ReviewDecision:
    """Result of reviewing a suspicious assignment."""

    action: str
    doc_id: str
    target_category_id: str | None = None
    reason: str = ""


def find_suspicious_singletons(categories: list[dict[str, Any]]) -> list[str]:
    """Return category ids that are singleton candidates for merge review."""
    if len(categories) <= 3:
        return []
    return [c["id"] for c in categories if int(c.get("doc_count") or 0) == 1]


def review_pending_by_similarity(
    pending: dict[str, np.ndarray],
    categories: list[dict[str, Any]],
    threshold: float,
) -> dict[str, AssignmentResult | None]:
    """Local review fallback: retry pending docs with a lower or explicit threshold."""
    return {
        doc_id: assign_to_existing_category(vector, categories, threshold)
        for doc_id, vector in pending.items()
    }


_DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "travel-policy": {"差旅", "出差", "住宿", "报销", "交通", "机票", "酒店", "票据"},
    "it-operations": {"vpn", "it", "设备", "远程", "接入", "账号", "电脑", "申请"},
    "health-benefits": {"体检", "健康", "福利", "医疗", "保险"},
    "national-standards": {"国家标准", "gb", "电子凭证", "会计档案", "标准", "规范"},
    "pageindex-docs": {"pageindex", "github", "索引", "检索", "rag"},
}


def _doc_text(docs: list[dict[str, Any]]) -> str:
    return " ".join(
        str(doc.get("title") or "") + " " + str(doc.get("source_path") or "")
        for doc in docs
    ).lower()


def _domain_for_docs(docs: list[dict[str, Any]]) -> str | None:
    text = _doc_text(docs)
    scores = {
        domain: sum(1 for keyword in keywords if keyword.lower() in text)
        for domain, keywords in _DOMAIN_KEYWORDS.items()
    }
    best_domain, best_score = max(scores.items(), key=lambda item: item[1])
    return best_domain if best_score > 0 else None


def merge_singleton_categories_by_keywords(
    categories: list[dict[str, Any]],
    category_docs: dict[str, list[dict[str, Any]]],
) -> list[list[str]]:
    """Group singleton categories that clearly belong to the same broad domain.

    This is a deterministic fallback for small corpora when embedding geometry
    splits every document into its own cluster before LLM review can help.
    """
    grouped: dict[str, list[str]] = {}
    passthrough: list[list[str]] = []

    for category in categories:
        cid = category["id"]
        docs = category_docs.get(cid, [])
        if int(category.get("doc_count") or len(docs)) != 1:
            passthrough.append([cid])
            continue
        domain = _domain_for_docs(docs)
        if domain is None:
            passthrough.append([cid])
        else:
            grouped.setdefault(domain, []).append(cid)

    result = [ids for ids in grouped.values()]
    result.extend(passthrough)
    return result

