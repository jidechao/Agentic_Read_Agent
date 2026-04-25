"""Cluster quality reporting."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.clustering.coarse import cosine_similarity


def _format_similarity(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def build_cluster_report(registry: Any, suspicious_threshold: float = 0.4) -> str:
    """Build a Markdown report for current stable categories."""
    categories = registry.list_categories()
    lines = ["# 聚类质量报告", ""]
    if not categories:
        lines.append("暂无稳定类目。")
        return "\n".join(lines) + "\n"

    suspicious: list[tuple[str, str, float | None]] = []
    all_centroids = [
        c["centroid"] for c in categories if c.get("centroid") is not None
    ]

    for category in categories:
        docs = registry.list_category_documents(category["id"])
        centroid = category.get("centroid")
        similarities: list[float] = []
        lines.append(
            f"## {category['display_name'] or category['canonical_name']} "
            f"({category['canonical_name']})"
        )
        lines.append(category.get("description") or "")
        lines.append(f"- 文档数：{len(docs)}")

        for doc in docs:
            embedding = registry.get_embedding(doc["id"])
            sim = None
            if embedding is not None and centroid is not None:
                sim = cosine_similarity(embedding, centroid)
                similarities.append(sim)
                if sim < suspicious_threshold:
                    suspicious.append((doc["source_path"], category["canonical_name"], sim))
            assignment_sim = doc.get("similarity")
            score_text = _format_similarity(sim if sim is not None else assignment_sim)
            lines.append(
                f"- {doc['source_path']}: {doc.get('title') or '未命名'} "
                f"(sim={score_text}, by={doc.get('assigned_by', 'unknown')})"
            )

        avg_intra = float(np.mean(similarities)) if similarities else None
        lines.append(f"- 簇内平均相似度：{_format_similarity(avg_intra)}")

        inter_scores = []
        if centroid is not None:
            for other in all_centroids:
                if other is centroid:
                    continue
                inter_scores.append(cosine_similarity(centroid, other))
        avg_inter = float(np.mean(inter_scores)) if inter_scores else None
        lines.append(f"- 与其他簇平均相似度：{_format_similarity(avg_inter)}")
        lines.append("")

    lines.append("## 可疑文档")
    if suspicious:
        for source_path, category_name, sim in suspicious:
            lines.append(f"- {source_path} in {category_name}: sim={_format_similarity(sim)}")
    else:
        lines.append("暂无。")

    return "\n".join(lines) + "\n"

