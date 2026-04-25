"""Category naming helpers."""
from __future__ import annotations

import json
import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer


def slugify_name(text: str, fallback: str = "cluster-default") -> str:
    """Convert model output or titles into a stable folder-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", text.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or fallback


def top_tfidf_keywords(texts: list[str], limit: int = 10) -> list[str]:
    """Extract lightweight TF-IDF keywords without adding tokenizer deps."""
    cleaned = [t for t in texts if t.strip()]
    if not cleaned:
        return []
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b[\w\u4e00-\u9fff]{2,}\b",
        max_features=max(limit * 3, limit),
    )
    try:
        matrix = vectorizer.fit_transform(cleaned)
    except ValueError:
        return []
    scores = matrix.sum(axis=0).A1
    names = vectorizer.get_feature_names_out()
    ranked = sorted(zip(names, scores), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked[:limit]]


def fallback_category_name(titles: list[str], summaries: list[str]) -> tuple[str, str, str]:
    """Produce a deterministic category name when LLM naming is unavailable."""
    keywords = top_tfidf_keywords(titles + summaries, limit=3)
    base = "-".join(keywords[:2]) if keywords else (titles[0] if titles else "knowledge")
    canonical = slugify_name(base, fallback="knowledge")
    display = " / ".join(keywords[:2]) if keywords else canonical
    description = "包含" + "、".join((titles or keywords)[:3]) + "等相关文档"
    return canonical, display, description


def name_category_with_llm(
    client: Any,
    model: str,
    titles: list[str],
    summaries: list[str],
) -> tuple[str, str, str]:
    """Ask the LLM to name a category, falling back to deterministic naming."""
    keywords = top_tfidf_keywords(titles + summaries)
    prompt = (
        "请为以下企业知识库分类命名。要求：\n"
        "1. canonical_name 使用 1-3 个英文词，小写，连字符连接；\n"
        "2. display_name 使用简短中文；\n"
        "3. description 用一句中文描述范围，避免过窄。\n\n"
        f"标题：{json.dumps(titles, ensure_ascii=False)}\n"
        f"摘要：{json.dumps(summaries[:5], ensure_ascii=False)}\n"
        f"关键词：{', '.join(keywords)}\n\n"
        '严格返回 JSON：{"canonical_name":"...","display_name":"...","description":"..."}'
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=240,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
        canonical = slugify_name(result["canonical_name"])
        return canonical, result.get("display_name") or canonical, result["description"]
    except Exception:
        return fallback_category_name(titles, summaries)

