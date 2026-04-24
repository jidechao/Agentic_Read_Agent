"""Incremental compilation engine.

Orchestrates: scan data dir -> detect changes -> ingest -> classify -> embed -> cluster.
Only processes documents that are new or changed since the last successful compile.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import src.config as cfg
from src.classifier import DocumentClassifier
from src.ingester import DocumentIngester, IngestResult
from src.registry import KnowledgeRegistry


# ── Helpers ───────────────────────────────────────────────────────────────


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

            for fi in changes["to_process"]:
                self.process_document(fi)

            for doc_id in changes["deleted"]:
                self.registry.delete_document(doc_id)

            num_clusters = 0
            if changes["to_process"] or changes["deleted"]:
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
                docs_processed=len(changes["to_process"]),
                docs_skipped=len(changes["skipped"]),
            )

            return CompileResult(
                docs_processed=len(changes["to_process"]),
                docs_skipped=len(changes["skipped"]),
                docs_deleted=len(changes["deleted"]),
                clusters_created=num_clusters,
            )
        except Exception as e:
            self.registry.complete_compile_run(run_id, error_message=str(e))
            raise

    # ── Scan / change detection ───────────────────────────────────────────

    def scan_data_dir(
        self, force: bool = False, data_dir: Path | None = None
    ) -> dict[str, list]:
        """Scan data/ dir recursively.

        Returns dict with to_process, skipped, and deleted lists.
        """
        scan_dir = data_dir or cfg.DATA_DIR
        supported_exts = {".pdf", ".md", ".markdown", ".docx", ".doc", ".html", ".htm"}

        to_process: list[FileInfo] = []
        skipped: list[FileInfo] = []
        seen_hashes: set[str] = set()

        for fpath in scan_dir.rglob("*"):
            if not fpath.is_file() or fpath.suffix.lower() not in supported_exts:
                continue
            content_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
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

        # Detect deleted: docs in registry not found in current scan
        all_docs = self.registry.list_documents()
        deleted: list[str] = []
        for doc in all_docs:
            if doc["content_hash"] not in seen_hashes:
                deleted.append(doc["id"])

        return {"to_process": to_process, "skipped": skipped, "deleted": deleted}

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
                embed_text = self._build_embed_text(result)
                vectors = self._get_embeddings([embed_text])
                self.registry.cache_embedding(doc_id, vectors[0])

            # If long, index via PageIndex
            if tier == "long":
                self._index_long_document(doc_id, fi.path, result)

            self.registry.update_document_status(doc_id, "embedded")
        except Exception as e:
            self.registry.update_document_status(doc_id, "error", error_message=str(e))

    # ── Embedding helpers ──────────────────────────────────────────────────

    @staticmethod
    def _build_embed_text(result: IngestResult) -> str:
        """Build embedding input: title + headings + content excerpt.

        Title and headings give topic structure, content excerpt adds
        domain-specific vocabulary for better cluster separation.
        """
        parts = []
        if result.title:
            parts.append(result.title)
        if result.headings:
            parts.extend(h.text for h in result.headings if h.text.strip())
        # Add first ~300 chars of body for domain vocabulary
        body = result.text[:300].strip()
        if body:
            parts.append(body)
        return " ".join(parts) if parts else result.text[:500]

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
                    np.array(item.embedding, dtype=np.float32)
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
        """KMeans clustering + LLM naming. Returns number of clusters.

        Uses title+heading embeddings (focused semantics) with auto-K
        selection via silhouette score.
        """
        docs = self.registry.list_documents(status="embedded")
        if not docs:
            return 0

        vectors: list[np.ndarray] = []
        doc_ids: list[str] = []
        for doc in docs:
            emb = self.registry.get_embedding(doc["id"])
            if emb is not None:
                vectors.append(emb)
                doc_ids.append(doc["id"])

        n = len(vectors)
        if n < 2:
            return 0

        matrix = np.array(vectors)
        k = self._find_optimal_k(matrix, n)
        print(f"自动选择 K={k} (共 {n} 个文档)")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(matrix)

        clusters = []
        for cluster_id in range(k):
            indices = [i for i, label in enumerate(labels) if label == cluster_id]
            titles = [
                docs[i]["title"] or docs[i]["source_path"] for i in indices
            ]
            cids = [doc_ids[i] for i in indices]
            name, description = self._name_cluster(titles)
            clusters.append(
                {"name": name, "description": description, "doc_ids": cids}
            )

        self.registry.save_clusters(clusters, run_id)
        return len(clusters)

    @staticmethod
    def _find_optimal_k(matrix: np.ndarray, n: int) -> int:
        """Find optimal K via silhouette score with granularity guard.

        When all scores are low (embeddings are close), prefer more clusters
        to avoid dumping unrelated docs into the same group.
        """
        if n <= 2:
            return 2
        max_k = min(n - 1, max(2, int(math.sqrt(n) * 2)))
        if max_k < 2:
            return 2

        scores: dict[int, float] = {}
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(matrix)
            if len(set(labels)) < 2:
                continue
            scores[k] = float(silhouette_score(matrix, labels))

        if not scores:
            return 2

        best_k = max(scores, key=scores.get)
        best_score = scores[best_k]

        # Granularity guard: if best score is low, embeddings are too similar.
        # Prefer finer granularity so each cluster is smaller and more focused.
        if best_score < 0.15:
            threshold = best_score * 0.8
            for k in sorted(scores.keys(), reverse=True):
                if scores[k] >= threshold:
                    return k

        return best_k

    def _name_cluster(self, doc_titles: list[str]) -> tuple[str, str]:
        """Call LLM to name a cluster based on its document titles."""
        prompt = (
            "以下是同一分类的企业知识文档标题列表：\n"
            + "\n".join(f"- {t}" for t in doc_titles)
            + "\n\n请用一个简短的英文短语作为分类目录名（小写，用连字符连接，如 it-hr），"
            "再写一句中文描述该分类。\n"
            '请严格按以下 JSON 格式返回：{"name": "xxx", "description": "xxx"}'
        )
        try:
            response = self._sf_client.chat.completions.create(
                model=cfg.CLUSTER_NAMING_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text.strip())
            return result["name"], result["description"]
        except (json.JSONDecodeError, KeyError):
            return "cluster-default", "默认分类"

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
