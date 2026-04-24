"""Atomic materialization — writes compiled_library/ output via tmp→bak→rename."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import src.config as cfg

logger = logging.getLogger(__name__)


class Materializer:
    """Atomically writes compiled_library/ output using tmp→bak→rename pattern."""

    def materialize(
        self,
        compile_run_id: int,
        registry: Any,
        short_doc_texts: dict[str, str] | None = None,
    ) -> None:
        """Atomically write compiled_library/ output.

        Args:
            compile_run_id: The compile run to materialize.
            registry: Knowledge registry for document/cluster queries.
            short_doc_texts: Mapping of doc_id → full text for short documents.
        """
        compiled_dir = cfg.COMPILED_DIR
        tmp_dir = compiled_dir.parent / (compiled_dir.name + ".tmp")
        bak_dir = compiled_dir.parent / (compiled_dir.name + ".bak")

        # Clean tmp from any previous failed run
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Gather data from registry
        docs = registry.list_documents()
        clusters = self._get_clusters(registry, compile_run_id)

        # Write files into tmp_dir
        self._write_skill_md(tmp_dir, clusters)
        self._write_short_docs_db(tmp_dir, docs, short_doc_texts or {})
        self._write_cluster_indexes(tmp_dir, clusters, docs)

        # Preserve pageindex_cache from current compiled_library/
        pi_cache = compiled_dir / "pageindex_cache"
        if pi_cache.exists():
            shutil.copytree(pi_cache, tmp_dir / "pageindex_cache", dirs_exist_ok=True)

        # Atomic swap
        if bak_dir.exists():
            shutil.rmtree(bak_dir)
        if compiled_dir.exists():
            compiled_dir.rename(bak_dir)
        tmp_dir.rename(compiled_dir)

        logger.info("原子物化完成: %s", compiled_dir)

    # ── Internal helpers ────────────────────────────────────────────────

    def _get_clusters(self, registry: Any, compile_run_id: int) -> list[dict[str, Any]]:
        """Get clusters with their documents from registry."""
        conn = registry._get_conn()
        rows = conn.execute(
            """SELECT c.id, c.name, c.description, dc.doc_id
               FROM clusters c
               LEFT JOIN document_clusters dc ON c.id = dc.cluster_id
               WHERE c.compile_run_id = ?
               ORDER BY c.id""",
            (compile_run_id,),
        ).fetchall()

        clusters_map: dict[int, dict[str, Any]] = {}
        for row in rows:
            cid = row[0]
            if cid not in clusters_map:
                clusters_map[cid] = {
                    "id": cid,
                    "name": row[1],
                    "description": row[2],
                    "doc_ids": [],
                }
            if row[3]:
                clusters_map[cid]["doc_ids"].append(row[3])

        return list(clusters_map.values())

    def _write_skill_md(self, tmp_dir: Path, clusters: list[dict[str, Any]]) -> None:
        """Write SKILL.md top-level directory index."""
        lines = ["# 知识库总目录", ""]
        for cluster in clusters:
            lines.append(f"## [{cluster['name']}]({cluster['name']}/INDEX.md)")
            lines.append(cluster["description"])
            lines.append("")
        (tmp_dir / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_short_docs_db(
        self,
        tmp_dir: Path,
        docs: list[dict[str, Any]],
        short_doc_texts: dict[str, str],
    ) -> None:
        """Write short_docs_db.json with full text content."""
        short_docs = {}
        for doc in docs:
            if doc["tier"] != "short" or doc["status"] not in ("embedded", "compiled"):
                continue
            content = short_doc_texts.get(doc["id"], "")
            if not content:
                continue
            short_docs[doc["id"]] = {
                "doc_id": doc["id"],
                "title": doc["title"],
                "source_path": doc["source_path"],
                "content": content,
            }
        if short_docs:
            (tmp_dir / "short_docs_db.json").write_text(
                json.dumps(short_docs, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _write_cluster_indexes(
        self,
        tmp_dir: Path,
        clusters: list[dict[str, Any]],
        docs: list[dict[str, Any]],
    ) -> None:
        """Write INDEX.md for each cluster with source info, grouped by tier."""
        docs_by_id = {doc["id"]: doc for doc in docs}

        for cluster in clusters:
            cluster_dir = tmp_dir / cluster["name"]
            cluster_dir.mkdir(parents=True, exist_ok=True)

            lines = [f"# {cluster['name']}", ""]

            # Separate short and long docs for clear navigation
            short_docs = []
            long_docs = []
            for doc_id in cluster["doc_ids"]:
                doc = docs_by_id.get(doc_id)
                if not doc:
                    continue
                if doc["tier"] == "short":
                    short_docs.append(doc)
                else:
                    long_docs.append(doc)

            if short_docs:
                lines.append("## 短文档（直接检索）")
                for doc in short_docs:
                    source = f" (来源: {doc['source_path']})" if doc.get("source_path") else ""
                    lines.append(
                        f"- {doc['id']}: {doc.get('title', '未命名')}{source}"
                    )
                lines.append("")

            if long_docs:
                lines.append("## 长文档（PageIndex 导航）")
                for doc in long_docs:
                    pid = doc.get("pageindex_id") or doc["id"]
                    pi_note = ", PageIndex索引" if doc.get("pageindex_id") else ""
                    lines.append(
                        f"- {pid}: {doc.get('title', '未命名')}"
                        f" (来源: {doc.get('source_path', '')}{pi_note})"
                    )
                lines.append("")

            (cluster_dir / "INDEX.md").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
