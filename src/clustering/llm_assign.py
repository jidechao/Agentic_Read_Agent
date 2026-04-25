"""Document-to-category assignment helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.clustering.coarse import cosine_similarity, l2_normalize


@dataclass(frozen=True)
class AssignmentResult:
    """A single category assignment decision."""

    category_id: str
    similarity: float
    confidence: float
    assigned_by: str = "embedding"


def assign_to_existing_category(
    vector: np.ndarray,
    categories: list[dict[str, Any]],
    threshold: float,
) -> AssignmentResult | None:
    """Assign a vector to the nearest existing category if confidence is high."""
    if not categories:
        return None

    normalized = l2_normalize(vector)
    best: tuple[dict[str, Any], float] | None = None
    for category in categories:
        centroid = category.get("centroid")
        if centroid is None:
            continue
        score = cosine_similarity(normalized, centroid)
        if best is None or score > best[1]:
            best = (category, score)

    if best is None or best[1] < threshold:
        return None

    return AssignmentResult(
        category_id=best[0]["id"],
        similarity=best[1],
        confidence=best[1],
        assigned_by="embedding",
    )


def reconcile_assignments(
    embedding_result: AssignmentResult | None,
    llm_result: AssignmentResult | None,
) -> AssignmentResult | None:
    """Choose a final assignment from embedding and LLM decisions.

    Agreement keeps the stronger confidence. Disagreement prefers the LLM result
    but caps confidence so review/reporting can surface the uncertainty.
    """
    if embedding_result is None:
        return llm_result
    if llm_result is None:
        return embedding_result
    if embedding_result.category_id == llm_result.category_id:
        confidence = max(embedding_result.confidence, llm_result.confidence)
        similarity = max(embedding_result.similarity, llm_result.similarity)
        return AssignmentResult(
            category_id=embedding_result.category_id,
            similarity=similarity,
            confidence=confidence,
            assigned_by="llm",
        )
    return AssignmentResult(
        category_id=llm_result.category_id,
        similarity=llm_result.similarity,
        confidence=min(llm_result.confidence, 0.69),
        assigned_by="llm",
    )

