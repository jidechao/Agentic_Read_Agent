"""SQLite-backed document registry with embedding cache and compile-run tracking."""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    source_path   TEXT NOT NULL,
    format        TEXT NOT NULL,
    title         TEXT,
    content_hash  TEXT NOT NULL,
    tier          TEXT CHECK(tier IN ('short', 'long', 'unclassified')) DEFAULT 'unclassified',
    status        TEXT CHECK(status IN ('ingested', 'classified', 'embedded', 'compiled', 'error')) DEFAULT 'ingested',
    token_count   INTEGER,
    has_structure BOOLEAN DEFAULT FALSE,
    embedding     BLOB,
    pageindex_id  TEXT,
    source_page_hint TEXT,
    error_message TEXT,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS compile_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_type    TEXT,
    started_at      DATETIME NOT NULL,
    completed_at    DATETIME,
    status          TEXT CHECK(status IN ('running', 'completed', 'failed')) DEFAULT 'running',
    docs_processed  INTEGER DEFAULT 0,
    docs_skipped    INTEGER DEFAULT 0,
    error_message   TEXT
);

CREATE TABLE IF NOT EXISTS clusters (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    description   TEXT,
    compile_run_id INTEGER REFERENCES compile_runs(id)
);

CREATE TABLE IF NOT EXISTS document_clusters (
    doc_id      TEXT REFERENCES documents(id),
    cluster_id  INTEGER REFERENCES clusters(id),
    PRIMARY KEY (doc_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS categories (
    id             TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL UNIQUE,
    display_name   TEXT,
    description    TEXT,
    centroid       BLOB,
    doc_count      INTEGER DEFAULT 0,
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at     DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS document_category (
    doc_id      TEXT REFERENCES documents(id),
    category_id TEXT REFERENCES categories(id),
    similarity  REAL,
    assigned_by TEXT CHECK(assigned_by IN ('embedding', 'llm', 'manual')) DEFAULT 'embedding',
    confidence  REAL,
    PRIMARY KEY (doc_id, category_id)
);

CREATE TABLE IF NOT EXISTS topic_summaries (
    content_hash TEXT PRIMARY KEY,
    summary      TEXT NOT NULL,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_document_category_category_id ON document_category(category_id);
"""


_BOOL_COLUMNS = {"has_structure"}


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    """Convert a sqlite3.Row to a plain dict, or return None."""
    if row is None:
        return None
    d = dict(row)
    for col in _BOOL_COLUMNS:
        if col in d and d[col] is not None:
            d[col] = bool(d[col])
    return d


def _vector_to_blob(vector: np.ndarray | None) -> bytes | None:
    """Serialize a float32 vector for SQLite storage."""
    if vector is None:
        return None
    return vector.astype(np.float32).tobytes()


def _blob_to_vector(blob: bytes | None) -> np.ndarray | None:
    """Deserialize a float32 vector from SQLite storage."""
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy()


class KnowledgeRegistry:
    """Manages document state, embedding cache, and compile runs in SQLite."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        """Return a lazily-initialised connection with tables created."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(SCHEMA)
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── Document CRUD ──────────────────────────────────────────────────

    def register_document(
        self,
        source_path: str,
        format: str,
        content_hash: str,
        title: str | None = None,
    ) -> str:
        """Insert a new document and return its generated id."""
        doc_id = str(uuid.uuid4())
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO documents (id, source_path, format, content_hash, title)
               VALUES (?, ?, ?, ?, ?)""",
            (doc_id, source_path, format, content_hash, title),
        )
        conn.commit()
        return doc_id

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Return a document dict by id, or None if not found."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        return _row_to_dict(row)

    def update_document_status(
        self,
        doc_id: str,
        status: str,
        tier: str | None = None,
        token_count: int | None = None,
        has_structure: bool | None = None,
        pageindex_id: str | None = None,
        error_message: str | None = None,
        title: str | None = None,
    ) -> None:
        """Update the status and optional fields of a document."""
        set_clauses: list[str] = ["status = ?", "updated_at = ?"]
        values: list[Any] = [
            status,
            datetime.now(timezone.utc).isoformat(),
        ]

        optional: dict[str, Any] = {
            "tier": tier,
            "token_count": token_count,
            "has_structure": has_structure,
            "pageindex_id": pageindex_id,
            "error_message": error_message,
            "title": title,
        }
        for col, val in optional.items():
            if val is not None:
                set_clauses.append(f"{col} = ?")
                values.append(val)

        values.append(doc_id)
        conn = self._get_conn()
        conn.execute(
            f"UPDATE documents SET {', '.join(set_clauses)} WHERE id = ?",
            values,
        )
        conn.commit()

    def delete_document(self, doc_id: str) -> None:
        """Delete a document and its cluster associations."""
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM document_clusters WHERE doc_id = ?", (doc_id,)
        )
        conn.execute("DELETE FROM document_category WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()

    def find_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        """Find a document by its content hash."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM documents WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        return _row_to_dict(row)

    def find_by_pageindex_id(self, pageindex_id: str) -> dict[str, Any] | None:
        """Find a document by its PageIndex ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM documents WHERE pageindex_id = ?",
            (pageindex_id,),
        ).fetchone()
        return _row_to_dict(row)

    def find_deleted_doc_ids(self, seen_hashes: set[str]) -> list[str]:
        """Return IDs of compiled docs whose content_hash is not in seen_hashes.

        If seen_hashes is empty (empty data dir), all compiled doc IDs are returned.
        """
        conn = self._get_conn()
        query = "SELECT id FROM documents WHERE status = 'compiled'"
        values: list[str] = []

        if seen_hashes:
            # placeholders contain only '?' literals; values passed as bind params
            placeholders = ",".join("?" for _ in seen_hashes)
            query += f" AND content_hash NOT IN ({placeholders})"
            values = list(seen_hashes)

        rows = conn.execute(query, values).fetchall()
        return [row["id"] for row in rows]

    def list_documents(
        self,
        status: str | None = None,
        tier: str | None = None,
    ) -> list[dict[str, Any]]:
        """List documents, optionally filtered by status and/or tier."""
        clauses: list[str] = []
        values: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            values.append(status)
        if tier is not None:
            clauses.append("tier = ?")
            values.append(tier)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        conn = self._get_conn()
        rows = conn.execute(
            f"SELECT * FROM documents{where}", values
        ).fetchall()
        return [_row_to_dict(r) for r in rows]  # type: ignore[misc]

    # ── Embedding cache ────────────────────────────────────────────────

    def cache_embedding(self, doc_id: str, vector: np.ndarray) -> None:
        """Store a numpy embedding vector as a BLOB."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE documents SET embedding = ? WHERE id = ?",
            (vector.astype(np.float32).tobytes(), doc_id),
        )
        conn.commit()

    def get_embedding(self, doc_id: str) -> np.ndarray | None:
        """Retrieve the stored embedding vector, or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT embedding FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if row is None or row["embedding"] is None:
            return None
        return np.frombuffer(row["embedding"], dtype=np.float32).copy()

    # ── Topic summary cache ─────────────────────────────────────────────

    def cache_topic_summary(self, content_hash: str, summary: str) -> None:
        """Cache an LLM-generated topic summary by content hash."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO topic_summaries (content_hash, summary)
               VALUES (?, ?)
               ON CONFLICT(content_hash) DO UPDATE SET summary = excluded.summary""",
            (content_hash, summary),
        )
        conn.commit()

    def get_topic_summary(self, content_hash: str) -> str | None:
        """Return a cached topic summary by content hash."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT summary FROM topic_summaries WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        return None if row is None else row["summary"]

    # ── Compile runs ───────────────────────────────────────────────────

    def create_compile_run(self, trigger_type: str) -> int:
        """Create a new compile run record and return its id."""
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO compile_runs (trigger_type, started_at)
               VALUES (?, ?)""",
            (trigger_type, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_compile_run(self, run_id: int) -> dict[str, Any] | None:
        """Return a compile run dict by id."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM compile_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return _row_to_dict(row)

    def complete_compile_run(
        self,
        run_id: int,
        docs_processed: int = 0,
        docs_skipped: int = 0,
        error_message: str | None = None,
    ) -> None:
        """Mark a compile run as completed with stats."""
        conn = self._get_conn()
        conn.execute(
            """UPDATE compile_runs
               SET status = 'completed',
                   completed_at = ?,
                   docs_processed = ?,
                   docs_skipped = ?,
                   error_message = ?
               WHERE id = ?""",
            (
                datetime.now(timezone.utc).isoformat(),
                docs_processed,
                docs_skipped,
                error_message,
                run_id,
            ),
        )
        conn.commit()

    # ── Clusters ───────────────────────────────────────────────────────

    def save_clusters(
        self,
        clusters: list[dict[str, Any]],
        compile_run_id: int,
    ) -> list[int]:
        """Persist clusters and their document linkages. Return cluster ids."""
        conn = self._get_conn()
        cluster_ids: list[int] = []
        for cluster in clusters:
            cursor = conn.execute(
                """INSERT INTO clusters (name, description, compile_run_id)
                   VALUES (?, ?, ?)""",
                (
                    cluster["name"],
                    cluster.get("description"),
                    compile_run_id,
                ),
            )
            cid = cursor.lastrowid
            assert cid is not None
            cluster_ids.append(cid)

            for doc_id in cluster.get("doc_ids", []):
                conn.execute(
                    """INSERT INTO document_clusters (doc_id, cluster_id)
                       VALUES (?, ?)""",
                    (doc_id, cid),
                )
        conn.commit()
        return cluster_ids

    # ── Stable categories ───────────────────────────────────────────────

    def upsert_category(
        self,
        canonical_name: str,
        display_name: str,
        description: str,
        centroid: np.ndarray | None,
        doc_count: int,
        category_id: str | None = None,
    ) -> str:
        """Insert or update a stable category and return its id."""
        conn = self._get_conn()
        existing = conn.execute(
            "SELECT id FROM categories WHERE canonical_name = ?",
            (canonical_name,),
        ).fetchone()
        now = datetime.now(timezone.utc).isoformat()
        blob = _vector_to_blob(centroid)

        if existing:
            cid = existing["id"]
            conn.execute(
                """UPDATE categories
                   SET display_name = ?,
                       description = ?,
                       centroid = ?,
                       doc_count = ?,
                       updated_at = ?
                   WHERE id = ?""",
                (display_name, description, blob, doc_count, now, cid),
            )
        else:
            cid = category_id or str(uuid.uuid4())
            conn.execute(
                """INSERT INTO categories
                   (id, canonical_name, display_name, description, centroid, doc_count, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cid,
                    canonical_name,
                    display_name,
                    description,
                    blob,
                    doc_count,
                    now,
                    now,
                ),
            )
        conn.commit()
        return cid

    def clear_categories(self) -> None:
        """Remove stable categories and their document assignments."""
        conn = self._get_conn()
        conn.execute("DELETE FROM document_category")
        conn.execute("DELETE FROM categories")
        conn.commit()

    def list_categories(self) -> list[dict[str, Any]]:
        """Return all stable categories ordered by creation time."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM categories ORDER BY created_at, canonical_name"
        ).fetchall()
        categories: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["centroid"] = _blob_to_vector(item.get("centroid"))
            categories.append(item)
        return categories

    def get_category_centroid(self, category_id: str) -> np.ndarray | None:
        """Return the centroid for a category."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT centroid FROM categories WHERE id = ?", (category_id,)
        ).fetchone()
        if row is None:
            return None
        return _blob_to_vector(row["centroid"])

    def assign_document_category(
        self,
        doc_id: str,
        category_id: str,
        similarity: float,
        assigned_by: str,
        confidence: float,
    ) -> None:
        """Persist one document-category assignment."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO document_category
               (doc_id, category_id, similarity, assigned_by, confidence)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(doc_id, category_id) DO UPDATE SET
                   similarity = excluded.similarity,
                   assigned_by = excluded.assigned_by,
                   confidence = excluded.confidence""",
            (doc_id, category_id, similarity, assigned_by, confidence),
        )
        conn.commit()

    def replace_document_category(
        self,
        doc_id: str,
        category_id: str,
        similarity: float,
        assigned_by: str,
        confidence: float,
    ) -> None:
        """Replace all existing category assignments for a document."""
        conn = self._get_conn()
        conn.execute("DELETE FROM document_category WHERE doc_id = ?", (doc_id,))
        conn.execute(
            """INSERT INTO document_category
               (doc_id, category_id, similarity, assigned_by, confidence)
               VALUES (?, ?, ?, ?, ?)""",
            (doc_id, category_id, similarity, assigned_by, confidence),
        )
        conn.commit()

    def list_category_documents(self, category_id: str) -> list[dict[str, Any]]:
        """Return documents assigned to a category with assignment metadata."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT d.*, dc.similarity, dc.assigned_by, dc.confidence
               FROM document_category dc
               JOIN documents d ON d.id = dc.doc_id
               WHERE dc.category_id = ?
               ORDER BY d.source_path""",
            (category_id,),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]  # type: ignore[misc]

    def list_document_categories(self) -> list[dict[str, Any]]:
        """Return all document-category assignments with document and category data."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT dc.doc_id,
                      dc.category_id,
                      dc.similarity,
                      dc.assigned_by,
                      dc.confidence,
                      d.source_path,
                      d.title,
                      d.tier,
                      c.canonical_name,
                      c.display_name,
                      c.description
               FROM document_category dc
               JOIN documents d ON d.id = dc.doc_id
               JOIN categories c ON c.id = dc.category_id
               ORDER BY c.canonical_name, d.source_path"""
        ).fetchall()
        return [dict(row) for row in rows]
