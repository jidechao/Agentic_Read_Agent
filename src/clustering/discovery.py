"""LLM-assisted category discovery from coarse embedding clusters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.clustering.coarse import centroid, cosine_similarity
from src.clustering.naming import name_category_with_llm


@dataclass(frozen=True)
class DiscoveredCategory:
    """A category proposed from one coarse cluster."""

    canonical_name: str
    display_name: str
    description: str
    doc_ids: list[str]
    centroid: np.ndarray


def group_by_label(doc_ids: list[str], labels: list[int]) -> dict[int, list[str]]:
    """Group document ids by cluster label."""
    grouped: dict[int, list[str]] = {}
    for doc_id, label in zip(doc_ids, labels):
        grouped.setdefault(label, []).append(doc_id)
    return grouped


def representative_doc_ids(
    doc_ids: list[str],
    vectors_by_doc_id: dict[str, np.ndarray],
    limit: int = 3,
) -> list[str]:
    """Return the ids closest to the local centroid."""
    vectors = [vectors_by_doc_id[doc_id] for doc_id in doc_ids]
    center = centroid(vectors)
    ranked = sorted(
        doc_ids,
        key=lambda doc_id: cosine_similarity(vectors_by_doc_id[doc_id], center),
        reverse=True,
    )
    return ranked[:limit]


def discover_categories(
    client: Any,
    model: str,
    docs_by_id: dict[str, dict[str, Any]],
    summaries: dict[str, str],
    vectors_by_doc_id: dict[str, np.ndarray],
    labels: list[int],
    doc_ids: list[str],
) -> list[DiscoveredCategory]:
    """Name each coarse cluster and return stable category candidates."""
    categories: list[DiscoveredCategory] = []
    for ids in group_by_label(doc_ids, labels).values():
        reps = representative_doc_ids(ids, vectors_by_doc_id)
        titles = [docs_by_id[doc_id].get("title") or docs_by_id[doc_id]["source_path"] for doc_id in reps]
        cluster_summaries = [summaries.get(doc_id, "") for doc_id in reps]
        canonical, display, description = name_category_with_llm(
            client, model, titles, cluster_summaries
        )
        categories.append(
            DiscoveredCategory(
                canonical_name=canonical,
                display_name=display,
                description=description,
                doc_ids=ids,
                centroid=centroid([vectors_by_doc_id[doc_id] for doc_id in ids]),
            )
        )
    return categories

