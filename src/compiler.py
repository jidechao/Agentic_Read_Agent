"""Incremental compilation engine.

Orchestrates: scan data dir -> detect changes -> ingest -> classify -> embed -> cluster.
Only processes documents that are new or changed since the last successful compile.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI
import src.config as cfg
from src.classifier import DocumentClassifier
from src.clustering.coarse import cosine_agglomerative, centroid, l2_normalize
from src.clustering.discovery import DiscoveredCategory, discover_categories
from src.clustering.llm_assign import assign_to_existing_category
from src.clustering.naming import name_category_with_llm
from src.clustering.review import merge_singleton_categories_by_keywords, review_pending_by_similarity
from src.ingester import DocumentIngester, IngestResult
from src.registry import KnowledgeRegistry


# ── Helpers ───────────────────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash using constant 64KB memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_ws(text: str) -> str:
    """Collapse consecutive whitespace and strip for fuzzy matching."""
    import re

    return re.sub(r"\s+", " ", text).strip()


def _find_heading(
    lookup: list[tuple[str, int, str]],
    counters: dict[str, int],
    norm_text: str,
) -> tuple[int, str] | None:
    """Find the next matching heading entry, advancing counter for duplicates."""
    idx = counters.get(norm_text, 0)
    for i in range(idx, len(lookup)):
        if lookup[i][0] == norm_text:
            counters[norm_text] = i + 1
            return lookup[i][1], lookup[i][2]
    return None


# ── Data classes ──────────────────────────────────────────────────────────


@dataclass
class FileInfo:
    """Lightweight descriptor for a file discovered during scanning."""

    path: Path
    content_hash: str
    relative_path: str


@dataclass
class ScanResult:
    """Typed result of a data directory scan."""

    to_process: list[FileInfo]
    skipped: list[FileInfo]
    deleted: list[str]


@dataclass
class CompileResult:
    """Summary returned after a compile run completes."""

    docs_processed: int
    docs_skipped: int
    docs_deleted: int
    clusters_created: int


# ── Compiler ─────────────────────────────────────────────────────────────


class KnowledgeCompiler:
    """Core compilation engine that orchestrates the full knowledge pipeline."""

    def __init__(self, registry: KnowledgeRegistry | None = None) -> None:
        self.registry = registry or KnowledgeRegistry(cfg.KNOWLEDGE_DB)
        self.ingester = DocumentIngester()
        self.classifier = DocumentClassifier()
        self._sf_client = OpenAI(
            api_key=cfg.SILICONFLOW_API_KEY,
            base_url=cfg.SILICONFLOW_BASE_URL,
        )
        self._short_doc_texts: dict[str, str] = {}  # doc_id → full text
        self._doc_summaries: dict[str, str] = {}  # doc_id → content summary

    # ── Main entry point ──────────────────────────────────────────────────

    def compile(
        self,
        trigger_type: str = "manual",
        force: bool = False,
        data_dir: Path | None = None,
    ) -> CompileResult:
        """Run a full compile cycle: scan, process, cluster."""
        run_id = self.registry.create_compile_run(trigger_type)
        try:
            changes = self.scan_data_dir(force, data_dir=data_dir)

            for fi in changes.to_process:
                self.process_document(fi)

            for doc_id in changes.deleted:
                self.registry.delete_document(doc_id)

            num_clusters = 0
            if changes.to_process or changes.deleted:
                if self.registry.list_categories() and not force and not changes.deleted:
                    num_clusters = self._incremental_assign_clusters(run_id)
                else:
                    num_clusters = self._compile_clusters(run_id)
                # Promote all embedded docs to compiled status
                for doc in self.registry.list_documents(status="embedded"):
                    self.registry.update_document_status(doc["id"], "compiled")
                # Atomically materialize compiled_library/
                from src.materializer import Materializer
                Materializer().materialize(
                    run_id, self.registry, self._short_doc_texts,
                    self._doc_summaries,
                )

            self.registry.complete_compile_run(
                run_id,
                docs_processed=len(changes.to_process),
                docs_skipped=len(changes.skipped),
            )

            return CompileResult(
                docs_processed=len(changes.to_process),
                docs_skipped=len(changes.skipped),
                docs_deleted=len(changes.deleted),
                clusters_created=num_clusters,
            )
        except Exception as e:
            self.registry.complete_compile_run(run_id, error_message=str(e))
            raise

    # ── Scan / change detection ───────────────────────────────────────────

    def scan_data_dir(
        self, force: bool = False, data_dir: Path | None = None
    ) -> ScanResult:
        """Scan data/ dir recursively.

        Returns ScanResult with to_process, skipped, and deleted lists.
        """
        scan_dir = data_dir or cfg.DATA_DIR
        supported_exts = {".pdf", ".md", ".markdown", ".docx", ".doc", ".html", ".htm"}

        to_process: list[FileInfo] = []
        skipped: list[FileInfo] = []
        seen_hashes: set[str] = set()

        for fpath in scan_dir.rglob("*"):
            if not fpath.is_file() or fpath.suffix.lower() not in supported_exts:
                continue
            content_hash = _sha256_file(fpath)
            seen_hashes.add(content_hash)
            rel_path = str(fpath.relative_to(scan_dir))
            fi = FileInfo(path=fpath, content_hash=content_hash, relative_path=rel_path)

            if force:
                to_process.append(fi)
                continue

            existing = self.registry.find_by_hash(content_hash)
            if existing and existing["status"] == "compiled":
                skipped.append(fi)
            else:
                to_process.append(fi)

        # Detect deleted: compiled docs whose hash is absent from current scan
        deleted = self.registry.find_deleted_doc_ids(seen_hashes)

        return ScanResult(to_process=to_process, skipped=skipped, deleted=deleted)

    # ── Single document pipeline ──────────────────────────────────────────

    def process_document(self, fi: FileInfo) -> None:
        """Process a single document: ingest -> classify -> embed."""
        # Check if already registered with this hash
        existing = self.registry.find_by_hash(fi.content_hash)
        if existing:
            doc_id = existing["id"]
            self.registry.update_document_status(doc_id, "ingested")  # reset
        else:
            doc_id = self.registry.register_document(
                source_path=fi.relative_path,
                format=fi.path.suffix.lstrip("."),
                content_hash=fi.content_hash,
            )

        try:
            # Ingest
            result = self.ingester.ingest(fi.path)

            # Classify
            tier = self.classifier.classify(result)
            tokens = self.classifier.estimate_tokens(result.text)
            has_structure = len(result.headings) >= 2
            self.registry.update_document_status(
                doc_id,
                "classified",
                tier=tier,
                token_count=tokens,
                has_structure=has_structure,
                title=result.title,
            )

            # Cache short doc text for materializer
            if tier == "short":
                self._short_doc_texts[doc_id] = result.text

            # Build summary for INDEX.md enrichment (all docs)
            self._doc_summaries[doc_id] = self._build_summary(result)

            # Embed (with cache check) — use title + headings for focused semantics
            cached = self.registry.get_embedding(doc_id)
            if cached is None:
                embed_text = self._build_embed_text(result, fi.content_hash)
                vectors = self._get_embeddings([embed_text])
                self.registry.cache_embedding(doc_id, vectors[0])

            # If long, index via PageIndex
            if tier == "long":
                self._index_long_document(doc_id, fi.path, result)

            self.registry.update_document_status(doc_id, "embedded")
        except Exception as e:
            self.registry.update_document_status(doc_id, "error", error_message=str(e))

    # ── Embedding helpers ──────────────────────────────────────────────────

    def _build_embed_text(self, result: IngestResult, content_hash: str) -> str:
        """Build embedding input: title + headings + content excerpt.

        A cached LLM topic summary carries the semantic signal while a small
        cleaned excerpt preserves domain vocabulary without PDF front-matter
        noise dominating the vector.
        """
        parts = []
        if result.title:
            parts.append(result.title)
        topic_summary = self._build_topic_summary(result, content_hash)
        if topic_summary:
            parts.append(topic_summary)
        if result.headings:
            parts.extend(h.text for h in result.headings[:8] if h.text.strip())
        body = self._strip_pdf_noise(result.text)[:200].strip()
        if body:
            parts.append(body)
        return " ".join(parts) if parts else result.text[:500]

    def _build_topic_summary(self, result: IngestResult, content_hash: str) -> str:
        """Return a cached 40-80 char topic summary, generating it if needed."""
        cached = self.registry.get_topic_summary(content_hash)
        if cached:
            return cached

        fallback = self._build_summary(result, max_chars=cfg.LLM_SUMMARY_MAX_CHARS)
        text = self._strip_pdf_noise(result.text)[:1500]
        prompt = (
            "请为以下企业知识库文档生成 40-80 字中文主题摘要，"
            "只描述主题、对象、关键概念，不要复述版权页、目录或页眉。\n\n"
            f"标题：{result.title}\n"
            f"正文片段：{text}\n"
        )
        try:
            response = self._sf_client.chat.completions.create(
                model=cfg.TOPIC_SUMMARY_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=160,
            )
            summary = response.choices[0].message.content.strip()
            summary = re.sub(r"\s+", " ", summary)[: cfg.LLM_SUMMARY_MAX_CHARS]
        except Exception:
            summary = fallback
        self.registry.cache_topic_summary(content_hash, summary)
        return summary

    @staticmethod
    def _strip_pdf_noise(text: str) -> str:
        """Remove common PDF front-matter lines before building semantic inputs."""
        noisy_patterns = (
            r"^\s*ICS\b",
            r"^\s*CCS\b",
            r"^\s*GB\b",
            r"GITHUB\s+TRENDING",
            r"^\s*Page\s+\d+\s*$",
            r"^\s*\d+\s*/\s*\d+\s*$",
            r"版权所有|Copyright",
            r"^\s*目\s*录\s*$",
        )
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if any(re.search(pattern, stripped, flags=re.IGNORECASE) for pattern in noisy_patterns):
                continue
            lines.append(stripped)
        return "\n".join(lines) if lines else text

    @staticmethod
    def _build_summary(result: IngestResult, max_chars: int = 120) -> str:
        """Build a short content summary from ingester data for INDEX.md.

        Pure extraction (no LLM calls): heading texts + first body line.
        """
        parts = []
        # Heading texts = key topics
        if result.headings:
            heading_texts = [h.text.strip() for h in result.headings if h.text.strip()]
            if heading_texts:
                parts.append("、".join(heading_texts[:6]))
        # First meaningful body line for domain vocabulary
        for line in result.text.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and len(stripped) > 5:
                if len(stripped) > max_chars:
                    stripped = stripped[:max_chars] + "..."
                parts.append(stripped)
                break
        return " | ".join(parts) if parts else result.text[:max_chars]

    # ── Embedding API (with retry) ────────────────────────────────────────

    def _get_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """Call SiliconFlow embedding API with exponential-backoff retry."""
        for attempt in range(3):
            try:
                response = self._sf_client.embeddings.create(
                    model=cfg.EMBEDDING_MODEL,
                    input=texts,
                    dimensions=cfg.EMBEDDING_DIMENSION,
                )
                return [
                    l2_normalize(np.array(item.embedding, dtype=np.float32))
                    for item in response.data
                ]
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        # Unreachable, but keeps the type checker happy
        return []  # pragma: no cover

    # ── Clustering ────────────────────────────────────────────────────────

    def _compile_clusters(self, run_id: int) -> int:
        """Cosine coarse clustering + LLM naming. Returns number of categories.

        Full compilation rebuilds stable categories from all embedded/compiled
        documents so old documents are not dropped when only one file changed.
        """
        old_categories = self.registry.list_categories()
        old_by_name = {
            category["canonical_name"]: category for category in old_categories
        }
        old_by_doc_id = {
            assignment["doc_id"]: old_by_name.get(assignment["canonical_name"])
            for assignment in self.registry.list_document_categories()
        }

        docs = self._list_clusterable_documents()
        if not docs:
            self.registry.clear_categories()
            return 0

        vectors: list[np.ndarray] = []
        doc_ids: list[str] = []
        for doc in docs:
            emb = self.registry.get_embedding(doc["id"])
            if emb is not None:
                vectors.append(emb)
                doc_ids.append(doc["id"])

        n = len(vectors)
        if n == 0:
            self.registry.clear_categories()
            return 0
        if n < 2:
            self.registry.clear_categories()
            doc_id = doc_ids[0]
            old_category = old_by_doc_id.get(doc_id)
            self._create_single_doc_category(
                doc_id,
                category_id=old_category["id"] if old_category else None,
                canonical_name=old_category["canonical_name"] if old_category else None,
                display_name=old_category.get("display_name") if old_category else None,
                description=old_category.get("description") if old_category else None,
            )
            return 1

        matrix = np.array(vectors)
        labels = cosine_agglomerative(matrix, cfg.CLUSTER_COSINE_THRESHOLD)
        print(f"余弦层次聚类生成 {len(set(labels))} 个候选类目 (共 {n} 个文档)")

        docs_by_id = {doc["id"]: doc for doc in docs}
        vectors_by_doc_id = dict(zip(doc_ids, vectors))
        summaries = {
            doc["id"]: self._summary_for_doc(doc)
            for doc in docs
        }

        discovered = self._merge_duplicate_discovered_categories(
            discover_categories(
                self._sf_client,
                cfg.CLUSTER_NAMING_MODEL,
                docs_by_id,
                summaries,
                vectors_by_doc_id,
                labels,
                doc_ids,
            )
        )
        discovered = self._reuse_old_category_names(
            discovered, old_categories, old_by_doc_id
        )
        discovered = self._merge_duplicate_discovered_categories(discovered)

        legacy_clusters: list[dict[str, object]] = []
        with self.registry.transaction():
            self.registry.clear_categories()
            for category in discovered:
                old_category = old_by_name.get(category.canonical_name)
                category_id = self.registry.upsert_category(
                    canonical_name=category.canonical_name,
                    display_name=category.display_name,
                    description=category.description,
                    centroid=category.centroid,
                    doc_count=len(category.doc_ids),
                    category_id=old_category["id"] if old_category else None,
                )
                for doc_id in category.doc_ids:
                    sim = float(
                        np.dot(
                            l2_normalize(vectors_by_doc_id[doc_id]),
                            category.centroid,
                        )
                    )
                    self.registry.replace_document_category(
                        doc_id,
                        category_id,
                        similarity=sim,
                        assigned_by="llm",
                        confidence=max(0.5, sim),
                    )
                legacy_clusters.append(
                    {
                        "name": category.canonical_name,
                        "description": category.description,
                        "doc_ids": category.doc_ids,
                    }
                )

        self._merge_singleton_categories(original_categories=old_categories)
        legacy_clusters = [
            {
                "name": category["canonical_name"],
                "description": category.get("description"),
                "doc_ids": [
                    doc["id"]
                    for doc in self.registry.list_category_documents(category["id"])
                ],
            }
            for category in self.registry.list_categories()
        ]
        with self.registry.transaction():
            self.registry.save_clusters(legacy_clusters, run_id)
        return len(self.registry.list_categories())

    def _incremental_assign_clusters(self, run_id: int) -> int:
        """Assign newly embedded documents to existing stable categories."""
        categories = self.registry.list_categories()
        pending: dict[str, np.ndarray] = {}
        legacy_clusters: dict[str, dict[str, object]] = {
            c["id"]: {
                "name": c["canonical_name"],
                "description": c.get("description"),
                "doc_ids": [],
            }
            for c in categories
        }

        for doc in self.registry.list_documents(status="embedded"):
            emb = self.registry.get_embedding(doc["id"])
            if emb is None:
                continue
            assignment = assign_to_existing_category(
                emb, categories, cfg.INCREMENTAL_ASSIGN_THRESHOLD
            )
            if assignment is None:
                pending[doc["id"]] = emb
                continue
            self.registry.replace_document_category(
                doc["id"],
                assignment.category_id,
                assignment.similarity,
                assignment.assigned_by,
                assignment.confidence,
            )
            legacy_clusters[assignment.category_id]["doc_ids"].append(doc["id"])

        if pending:
            if self._pending_requires_rebalance(len(pending)):
                return self._compile_clusters(run_id)
            reviewed = review_pending_by_similarity(
                pending,
                categories,
                threshold=max(0.35, cfg.INCREMENTAL_ASSIGN_THRESHOLD - 0.15),
            )
            for doc_id, assignment in reviewed.items():
                if assignment is None:
                    self._create_single_doc_category(doc_id)
                    continue
                self.registry.replace_document_category(
                    doc_id,
                    assignment.category_id,
                    assignment.similarity,
                    "llm",
                    min(assignment.confidence, 0.69),
                )
                legacy_clusters[assignment.category_id]["doc_ids"].append(doc_id)

        self._refresh_category_centroids()
        self.registry.save_clusters(list(legacy_clusters.values()), run_id)
        return len(self.registry.list_categories())

    def _list_clusterable_documents(self) -> list[dict[str, object]]:
        """Return documents that should participate in category materialization."""
        docs = []
        for status in ("embedded", "compiled"):
            docs.extend(self.registry.list_documents(status=status))
        return docs

    def _summary_for_doc(self, doc: dict[str, object]) -> str:
        """Return cached summary for a registry document."""
        doc_id = str(doc["id"])
        if doc_id in self._doc_summaries:
            return self._doc_summaries[doc_id]
        cached = self.registry.get_topic_summary(str(doc["content_hash"]))
        if cached:
            self._doc_summaries[doc_id] = cached
            return cached
        fallback = str(doc.get("title") or doc.get("source_path") or "")
        self._doc_summaries[doc_id] = fallback
        return fallback

    def _create_single_doc_category(
        self,
        doc_id: str,
        category_id: str | None = None,
        canonical_name: str | None = None,
        display_name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Create a category for a low-confidence document as a review fallback."""
        doc = self.registry.get_document(doc_id)
        emb = self.registry.get_embedding(doc_id)
        if doc is None or emb is None:
            return
        title = doc.get("title") or doc.get("source_path") or "knowledge"
        summary = self._summary_for_doc(doc)
        if canonical_name is None or display_name is None or description is None:
            generated = name_category_with_llm(
                self._sf_client, cfg.CLUSTER_NAMING_MODEL, [title], [summary]
            )
            canonical_name = canonical_name or generated[0]
            display_name = display_name or generated[1]
            description = description or generated[2]
        saved_category_id = self.registry.upsert_category(
            canonical_name,
            display_name,
            description,
            l2_normalize(emb),
            doc_count=1,
            category_id=category_id,
        )
        self.registry.replace_document_category(
            doc_id, saved_category_id, 1.0, "llm", 0.5
        )

    def _merge_duplicate_discovered_categories(
        self, categories: list[DiscoveredCategory]
    ) -> list[DiscoveredCategory]:
        """Merge discovered clusters that resolved to the same canonical name."""
        grouped: dict[str, list[DiscoveredCategory]] = {}
        for category in categories:
            grouped.setdefault(category.canonical_name, []).append(category)

        merged: list[DiscoveredCategory] = []
        for same_name in grouped.values():
            if len(same_name) == 1:
                merged.append(same_name[0])
                continue
            doc_ids: list[str] = []
            vectors: list[np.ndarray] = []
            for category in same_name:
                doc_ids.extend(category.doc_ids)
                vectors.append(category.centroid)
            first = same_name[0]
            merged.append(
                DiscoveredCategory(
                    canonical_name=first.canonical_name,
                    display_name=first.display_name,
                    description=first.description,
                    doc_ids=doc_ids,
                    centroid=centroid(vectors),
                )
            )
        return merged

    def _reuse_old_category_names(
        self,
        discovered: list[DiscoveredCategory],
        old_categories: list[dict[str, object]],
        old_by_doc_id: dict[str, dict[str, object] | None] | None = None,
    ) -> list[DiscoveredCategory]:
        """Reuse old category names when centroids still describe the same topic."""
        if not old_categories or not discovered:
            return discovered

        result = list(discovered)
        old_by_name = {
            str(old["canonical_name"]): old
            for old in old_categories
        }
        used_old: set[str] = set()
        used_new: set[int] = set()

        for new_idx, category in enumerate(result):
            old = old_by_name.get(category.canonical_name)
            if old is None:
                continue
            old_id = str(old["id"])
            used_new.add(new_idx)
            used_old.add(old_id)
            result[new_idx] = DiscoveredCategory(
                canonical_name=str(old["canonical_name"]),
                display_name=str(old.get("display_name") or old["canonical_name"]),
                description=str(old.get("description") or category.description),
                doc_ids=category.doc_ids,
                centroid=category.centroid,
            )

        pairs: list[tuple[float, int, dict[str, object]]] = []
        for new_idx, category in enumerate(result):
            if new_idx in used_new:
                continue
            new_centroid = l2_normalize(category.centroid)
            for old in old_categories:
                if str(old["id"]) in used_old:
                    continue
                old_centroid = old.get("centroid")
                if old_centroid is None:
                    continue
                sim = float(np.dot(new_centroid, l2_normalize(old_centroid)))
                pairs.append((sim, new_idx, old))
        pairs.sort(key=lambda pair: pair[0], reverse=True)

        for sim, new_idx, old in pairs:
            if sim < cfg.CATEGORY_REUSE_THRESHOLD:
                break
            old_id = str(old["id"])
            if new_idx in used_new or old_id in used_old:
                continue
            used_new.add(new_idx)
            used_old.add(old_id)
            category = result[new_idx]
            result[new_idx] = DiscoveredCategory(
                canonical_name=str(old["canonical_name"]),
                display_name=str(old.get("display_name") or old["canonical_name"]),
                description=str(old.get("description") or category.description),
                doc_ids=category.doc_ids,
                centroid=category.centroid,
            )

        if not old_by_doc_id:
            return result

        for new_idx, category in enumerate(result):
            if new_idx in used_new:
                continue
            counts: dict[str, tuple[int, dict[str, object]]] = {}
            for doc_id in category.doc_ids:
                old = old_by_doc_id.get(doc_id)
                if old is None:
                    continue
                old_id = str(old["id"])
                if old_id in used_old:
                    continue
                count, _ = counts.get(old_id, (0, old))
                counts[old_id] = (count + 1, old)
            if not counts:
                continue
            _, (overlap_count, old) = max(
                counts.items(), key=lambda item: item[1][0]
            )
            if overlap_count / len(category.doc_ids) < 0.5:
                continue
            used_new.add(new_idx)
            used_old.add(str(old["id"]))
            result[new_idx] = DiscoveredCategory(
                canonical_name=str(old["canonical_name"]),
                display_name=str(old.get("display_name") or old["canonical_name"]),
                description=str(old.get("description") or category.description),
                doc_ids=category.doc_ids,
                centroid=category.centroid,
            )
        return result

    def _pending_requires_rebalance(self, pending_count: int) -> bool:
        """Return whether pending volume should trigger full rebalance."""
        total = len(self._list_clusterable_documents())
        if total == 0:
            return False
        if pending_count >= cfg.PENDING_REVIEW_REBALANCE_ABSOLUTE:
            return True
        if pending_count < cfg.PENDING_REVIEW_REBALANCE_MIN:
            return False
        return pending_count / total >= cfg.PENDING_REVIEW_REBALANCE_RATIO

    def _refresh_category_centroids(self) -> None:
        """Recompute category centroids and doc counts after assignments."""
        for category in self.registry.list_categories():
            docs = self.registry.list_category_documents(category["id"])
            vectors = [
                emb for doc in docs
                if (emb := self.registry.get_embedding(doc["id"])) is not None
            ]
            if not vectors:
                continue
            self.registry.upsert_category(
                category["canonical_name"],
                category.get("display_name") or category["canonical_name"],
                category.get("description") or "",
                centroid(vectors),
                doc_count=len(docs),
                category_id=category["id"],
            )

    def _merge_singleton_categories(
        self, original_categories: list[dict[str, object]] | None = None
    ) -> None:
        """Merge obvious singleton categories produced by over-fine coarse clustering."""
        categories = self.registry.list_categories()
        docs_by_category = {
            category["id"]: self.registry.list_category_documents(category["id"])
            for category in categories
        }
        groups = merge_singleton_categories_by_keywords(categories, docs_by_category)
        if all(len(group) == 1 for group in groups):
            return

        old_by_id = {category["id"]: category for category in categories}

        plans: list[
            tuple[list[str], list[dict[str, object]], list[np.ndarray], str, str, str]
        ] = []
        for group in groups:
            docs: list[dict[str, object]] = []
            vectors: list[np.ndarray] = []
            for category_id in group:
                for doc in docs_by_category.get(category_id, []):
                    docs.append(doc)
                    emb = self.registry.get_embedding(doc["id"])
                    if emb is not None:
                        vectors.append(emb)

            canonical: str | None = None
            display: str | None = None
            description: str | None = None
            cluster_centroid = centroid(vectors) if vectors else None

            if cluster_centroid is not None and original_categories:
                best: dict[str, object] | None = None
                best_sim = -1.0
                for old in original_categories:
                    old_centroid = old.get("centroid")
                    if old_centroid is None:
                        continue
                    sim = float(
                        np.dot(
                            l2_normalize(cluster_centroid),
                            l2_normalize(old_centroid),
                        )
                    )
                    if sim > best_sim:
                        best_sim = sim
                        best = old
                if best is not None and best_sim >= cfg.CATEGORY_REUSE_THRESHOLD:
                    canonical = str(best["canonical_name"])
                    display = str(best.get("display_name") or canonical)
                    description = str(best.get("description") or "")

            if canonical is None and len(group) == 1:
                original = old_by_id[group[0]]
                canonical = str(original["canonical_name"])
                display = str(original.get("display_name") or canonical)
                description = str(original.get("description") or "")

            if canonical is None:
                titles = [
                    str(doc.get("title") or doc.get("source_path") or "")
                    for doc in docs
                ]
                summaries = [self._summary_for_doc(doc) for doc in docs]
                canonical, display, description = name_category_with_llm(
                    self._sf_client, cfg.CLUSTER_NAMING_MODEL, titles, summaries
                )

            plans.append((group, docs, vectors, canonical, display, description))

        with self.registry.transaction():
            self.registry.clear_categories()
            for group, docs, vectors, canonical, display, description in plans:
                reuse_id = old_by_id[group[0]]["id"]
                category_id = self.registry.upsert_category(
                    canonical,
                    display,
                    description,
                    centroid(vectors) if vectors else None,
                    doc_count=len(docs),
                    category_id=reuse_id,
                )
                for doc in docs:
                    emb = self.registry.get_embedding(doc["id"])
                    center = self.registry.get_category_centroid(category_id)
                    sim = (
                        float(np.dot(l2_normalize(emb), center))
                        if emb is not None and center is not None
                        else 0.0
                    )
                    self.registry.replace_document_category(
                        doc["id"], category_id, sim, "llm", max(0.5, sim)
                    )

    # ── Long document indexing ────────────────────────────────────────────

    def _index_long_document(
        self, doc_id: str, file_path: Path, result: IngestResult
    ) -> None:
        """Index a long document via PageIndex."""
        os.environ["OPENAI_API_KEY"] = cfg.SILICONFLOW_API_KEY
        os.environ["OPENAI_BASE_URL"] = cfg.SILICONFLOW_BASE_URL

        from pageindex import PageIndexClient

        litellm_model = f"openai/{cfg.PAGEINDEX_MODEL}"
        pi_client = PageIndexClient(
            workspace=cfg.PAGEINDEX_WORKSPACE, model=litellm_model
        )

        if file_path.suffix.lower() not in (".md", ".markdown"):
            # Non-md: reconstruct markdown with heading markers from ingester
            import tempfile

            md_content = self._reconstruct_markdown(result)
            with tempfile.NamedTemporaryFile(
                suffix=".md", mode="w", encoding="utf-8", delete=False
            ) as tmp:
                tmp.write(md_content)
                tmp_path = tmp.name
            try:
                pageindex_id = pi_client.index(tmp_path, mode="md")
            finally:
                os.unlink(tmp_path)
        else:
            pageindex_id = pi_client.index(str(file_path.resolve()), mode="md")

        self.registry.update_document_status(
            doc_id, "embedded", pageindex_id=pageindex_id
        )

    @staticmethod
    def _reconstruct_markdown(result: IngestResult) -> str:
        """Reconstruct proper markdown from ingester result.
        Inserts heading markers so PageIndex can build a tree structure.
        """
        if not result.headings:
            # No headings detected — wrap entire text as one section
            return f"# {result.title}\n\n{result.text}"

        # Build normalized lookup: (normalized_text -> level)
        # Use list of tuples to handle duplicate heading texts correctly
        heading_lookup: list[tuple[str, int, str]] = [
            (_normalize_ws(h.text), h.level, h.text) for h in result.headings
        ]
        # Track next heading index for each normalized text to handle duplicates
        heading_counters: dict[str, int] = {}

        lines = result.text.split("\n")
        output_lines = [f"# {result.title}", ""]
        for line in lines:
            stripped = line.strip()
            norm = _normalize_ws(stripped)
            # Find matching heading by normalized text, respecting order for duplicates
            match = _find_heading(heading_lookup, heading_counters, norm)
            if match is not None:
                level, original_text = match
                prefix = "#" * level
                output_lines.append(f"{prefix} {original_text}")
            else:
                output_lines.append(line)

        return "\n".join(output_lines)
