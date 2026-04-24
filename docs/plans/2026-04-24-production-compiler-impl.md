# 生产级知识库编译系统实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将硬编码的离线编译器改造为 SQLite-backed 的生产级知识库编译系统，支持多格式文档接入、自动分类、增量编译、原子输出和四种触发模式。

**Architecture:** SQLite 注册表管线 — 文档经过 Ingest → Classify → Embed → Compile → Materialize 五个阶段，每阶段通过 SQLite 记录状态实现断点续跑。编译产物格式与现有 Agent 完全兼容。

**Tech Stack:** Python 3.11, SQLite (内置), PyMuPDF (PDF), python-docx (DOCX), BeautifulSoup4 (HTML), mistune (Markdown), scikit-learn (聚类), openai (embedding), APScheduler (定时), FastAPI (API), watchdog (文件监控)

**设计文档:** `docs/plans/2026-04-24-production-compiler-design.md`

---

## Phase 1: 基础设施（注册表 + 配置）

### Task 1: 安装依赖 + 创建 requirements.txt

**Files:**
- Create: `requirements.txt`

**Step 1: 创建 requirements.txt**

```
openai>=2.32.0
openai-agents>=0.14.5
scikit-learn>=1.8.0
python-dotenv>=1.0.1
numpy>=2.4.0
PyMuPDF>=1.27.0
python-docx>=1.1.0
beautifulsoup4>=4.12.0
lxml>=5.0.0
mistune>=3.0.0
watchdog>=4.0.0
APScheduler>=3.10.0
fastapi>=0.115.0
uvicorn>=0.30.0
pytest>=8.0.0
```

**Step 2: 安装依赖**

Run: `pip install -r requirements.txt`
Expected: 所有包安装成功

**Step 3: 创建 tests 目录**

Run: `mkdir -p tests`

**Step 4: Commit**

```bash
git add requirements.txt tests/
git commit -m "chore: add requirements.txt and tests directory"
```

---

### Task 2: 扩展 config.py

**Files:**
- Modify: `src/config.py`

**Step 1: 写测试 — 验证新配置项存在且有默认值**

Create `tests/test_config.py`:

```python
"""验证配置模块的新增项存在且可访问。"""
import importlib
import sys
from pathlib import Path

# 确保 src 可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_config_has_classifier_settings():
    """分类器配置项存在且有合理默认值。"""
    # 需要在没有 .env 的情况下也能导入
    # 跳过 SILICONFLOW_API_KEY 检查
    import os
    os.environ.setdefault("SILICONFLOW_API_KEY", "test-key-for-config-test")

    import src.config as cfg

    assert hasattr(cfg, "CLASSIFIER_TOKEN_THRESHOLD")
    assert cfg.CLASSIFIER_TOKEN_THRESHOLD > 0
    assert hasattr(cfg, "CLASSIFIER_STRUCTURE_WEIGHT")
    assert 0 <= cfg.CLASSIFIER_STRUCTURE_WEIGHT <= 1
    assert hasattr(cfg, "CLASSIFIER_LENGTH_WEIGHT")
    assert 0 <= cfg.CLASSIFIER_LENGTH_WEIGHT <= 1
    assert hasattr(cfg, "CLASSIFIER_USE_EXACT_TOKENS")
    assert isinstance(cfg.CLASSIFIER_USE_EXACT_TOKENS, bool)


def test_config_has_registry_path():
    """知识库注册表路径配置存在。"""
    import os
    os.environ.setdefault("SILICONFLOW_API_KEY", "test-key-for-config-test")

    import src.config as cfg

    assert hasattr(cfg, "KNOWLEDGE_DB")
    assert isinstance(cfg.KNOWLEDGE_DB, Path)


def test_config_has_data_dir():
    """数据目录配置存在。"""
    import os
    os.environ.setdefault("SILICONFLOW_API_KEY", "test-key-for-config-test")

    import src.config as cfg

    assert hasattr(cfg, "DATA_DIR")
    assert isinstance(cfg.DATA_DIR, Path)
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `CLASSIFIER_TOKEN_THRESHOLD` 等属性不存在

**Step 3: 修改 config.py，在现有代码末尾追加新配置**

在 `src/config.py` 的 `CLUSTER_K` 之后追加：

```python
# ── 自动分类器配置 ──────────────────────────────────────────────────────
CLASSIFIER_TOKEN_THRESHOLD: int = int(os.environ.get("CLASSIFIER_TOKEN_THRESHOLD", "1000"))
CLASSIFIER_STRUCTURE_WEIGHT: float = float(os.environ.get("CLASSIFIER_STRUCTURE_WEIGHT", "0.4"))
CLASSIFIER_LENGTH_WEIGHT: float = float(os.environ.get("CLASSIFIER_LENGTH_WEIGHT", "0.6"))
CLASSIFIER_USE_EXACT_TOKENS: bool = os.environ.get("CLASSIFIER_USE_EXACT_TOKENS", "false").lower() == "true"

# ── 注册表路径 ──────────────────────────────────────────────────────────
KNOWLEDGE_DB: Path = PROJECT_ROOT / "knowledge.db"
```

**Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: extend config with classifier settings and registry path"
```

---

### Task 3: SQLite 注册表模块

**Files:**
- Create: `src/registry.py`
- Create: `tests/test_registry.py`

**Step 1: 写测试 — 注册表核心操作**

Create `tests/test_registry.py`:

```python
"""SQLite 文档注册表核心操作测试。"""
import os
import tempfile
from pathlib import Path

import pytest

# 确保可以导入
os.environ.setdefault("SILICONFLOW_API_KEY", "test-key")

from src.registry import KnowledgeRegistry


@pytest.fixture
def registry():
    """创建临时注册表。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        reg = KnowledgeRegistry(db_path)
        yield reg


def test_create_tables(registry):
    """表创建成功。"""
    conn = registry._get_conn()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {row[0] for row in tables}
    assert "documents" in table_names
    assert "compile_runs" in table_names
    assert "clusters" in table_names
    assert "document_clusters" in table_names


def test_register_document(registry):
    """注册文档并验证字段。"""
    doc_id = registry.register_document(
        source_path="data/test.pdf",
        format="pdf",
        content_hash="abc123",
        title="测试文档",
    )
    assert doc_id is not None

    doc = registry.get_document(doc_id)
    assert doc["source_path"] == "data/test.pdf"
    assert doc["format"] == "pdf"
    assert doc["status"] == "ingested"
    assert doc["content_hash"] == "abc123"
    assert doc["title"] == "测试文档"


def test_update_document_status(registry):
    """文档状态流转。"""
    doc_id = registry.register_document(
        source_path="data/test.md",
        format="md",
        content_hash="hash1",
    )
    registry.update_document_status(doc_id, "classified", tier="long")
    doc = registry.get_document(doc_id)
    assert doc["status"] == "classified"
    assert doc["tier"] == "long"


def test_set_document_error(registry):
    """错误状态记录。"""
    doc_id = registry.register_document(
        source_path="data/test.md",
        format="md",
        content_hash="hash1",
    )
    registry.update_document_status(doc_id, "error", error_message="API 超时")
    doc = registry.get_document(doc_id)
    assert doc["status"] == "error"
    assert doc["error_message"] == "API 超时"


def test_cache_embedding(registry):
    """向量缓存读写。"""
    import numpy as np

    doc_id = registry.register_document(
        source_path="data/test.md",
        format="md",
        content_hash="hash1",
    )
    vec = np.random.rand(1536).astype(np.float32)
    registry.cache_embedding(doc_id, vec)

    loaded = registry.get_embedding(doc_id)
    assert loaded is not None
    assert np.allclose(vec, loaded)


def test_find_by_hash(registry):
    """通过 hash 查找文档。"""
    registry.register_document("data/a.md", "md", "hash_x")
    registry.register_document("data/b.md", "md", "hash_y")

    found = registry.find_by_hash("hash_x")
    assert found is not None
    assert found["source_path"] == "data/a.md"

    not_found = registry.find_by_hash("hash_z")
    assert not_found is None


def test_delete_document(registry):
    """删除文档。"""
    doc_id = registry.register_document("data/test.md", "md", "hash1")
    registry.delete_document(doc_id)
    assert registry.get_document(doc_id) is None


def test_list_documents_by_status(registry):
    """按状态列出文档。"""
    id1 = registry.register_document("data/a.md", "md", "h1")
    id2 = registry.register_document("data/b.md", "md", "h2")
    registry.update_document_status(id1, "compiled")

    compiled = registry.list_documents(status="compiled")
    assert len(compiled) == 1
    assert compiled[0]["id"] == id1

    ingested = registry.list_documents(status="ingested")
    assert len(ingested) == 1


def test_create_compile_run(registry):
    """编译记录创建。"""
    run_id = registry.create_compile_run(trigger_type="manual")
    assert run_id is not None

    run = registry.get_compile_run(run_id)
    assert run["trigger_type"] == "manual"
    assert run["status"] == "running"


def test_complete_compile_run(registry):
    """编译记录完成。"""
    run_id = registry.create_compile_run("manual")
    registry.complete_compile_run(run_id, docs_processed=5, docs_skipped=2)

    run = registry.get_compile_run(run_id)
    assert run["status"] == "completed"
    assert run["docs_processed"] == 5
    assert run["docs_skipped"] == 2
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_registry.py -v`
Expected: FAIL — `src.registry` 模块不存在

**Step 3: 实现 registry.py**

Create `src/registry.py`:

```python
"""SQLite 文档注册表 — 追踪文档状态、embedding 缓存、编译历史。"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

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
"""


class KnowledgeRegistry:
    """SQLite-backed knowledge base document registry."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(SCHEMA)
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── 文档操作 ──────────────────────────────────────────────────────

    def register_document(
        self,
        source_path: str,
        format: str,
        content_hash: str,
        title: str | None = None,
    ) -> str:
        """注册文档，返回 doc_id。"""
        doc_id = str(uuid.uuid4())
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO documents (id, source_path, format, content_hash, title)
               VALUES (?, ?, ?, ?, ?)""",
            (doc_id, source_path, format, content_hash, title),
        )
        conn.commit()
        return doc_id

    def get_document(self, doc_id: str) -> dict | None:
        """获取文档记录。"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_document_status(
        self,
        doc_id: str,
        status: str,
        tier: str | None = None,
        token_count: int | None = None,
        has_structure: bool | None = None,
        pageindex_id: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """更新文档状态和可选字段。"""
        conn = self._get_conn()
        sets = ["status = ?", "updated_at = ?"]
        params: list = [status, datetime.now(timezone.utc).isoformat()]

        if tier is not None:
            sets.append("tier = ?")
            params.append(tier)
        if token_count is not None:
            sets.append("token_count = ?")
            params.append(token_count)
        if has_structure is not None:
            sets.append("has_structure = ?")
            params.append(has_structure)
        if pageindex_id is not None:
            sets.append("pageindex_id = ?")
            params.append(pageindex_id)
        if error_message is not None:
            sets.append("error_message = ?")
            params.append(error_message)

        params.append(doc_id)
        conn.execute(
            f"UPDATE documents SET {', '.join(sets)} WHERE id = ?", params
        )
        conn.commit()

    def delete_document(self, doc_id: str) -> None:
        """删除文档及关联记录。"""
        conn = self._get_conn()
        conn.execute("DELETE FROM document_clusters WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()

    def find_by_hash(self, content_hash: str) -> dict | None:
        """通过内容 hash 查找文档。"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return dict(row) if row else None

    def list_documents(
        self,
        status: str | None = None,
        tier: str | None = None,
    ) -> list[dict]:
        """列出文档，可按状态/tier过滤。"""
        conn = self._get_conn()
        query = "SELECT * FROM documents WHERE 1=1"
        params: list = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if tier:
            query += " AND tier = ?"
            params.append(tier)
        query += " ORDER BY created_at"
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ── Embedding 缓存 ──────────────────────────────────────────────

    def cache_embedding(self, doc_id: str, vector: np.ndarray) -> None:
        """缓存文档向量。"""
        conn = self._get_conn()
        conn.execute(
            "UPDATE documents SET embedding = ? WHERE id = ?",
            (vector.tobytes(), doc_id),
        )
        conn.commit()

    def get_embedding(self, doc_id: str) -> np.ndarray | None:
        """读取缓存的向量。"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT embedding FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if row and row["embedding"]:
            return np.frombuffer(row["embedding"], dtype=np.float32)
        return None

    # ── 编译记录 ──────────────────────────────────────────────────────

    def create_compile_run(self, trigger_type: str) -> int:
        """创建编译记录，返回 run_id。"""
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO compile_runs (trigger_type, started_at)
               VALUES (?, ?)""",
            (trigger_type, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return cursor.lastrowid

    def get_compile_run(self, run_id: int) -> dict | None:
        """获取编译记录。"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM compile_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def complete_compile_run(
        self,
        run_id: int,
        docs_processed: int = 0,
        docs_skipped: int = 0,
        error_message: str | None = None,
    ) -> None:
        """标记编译完成。"""
        status = "failed" if error_message else "completed"
        conn = self._get_conn()
        conn.execute(
            """UPDATE compile_runs
               SET completed_at = ?, status = ?, docs_processed = ?,
                   docs_skipped = ?, error_message = ?
               WHERE id = ?""",
            (
                datetime.now(timezone.utc).isoformat(),
                status,
                docs_processed,
                docs_skipped,
                error_message,
                run_id,
            ),
        )
        conn.commit()

    # ── 聚类结果 ──────────────────────────────────────────────────────

    def save_clusters(
        self, clusters: list[dict], compile_run_id: int
    ) -> list[int]:
        """保存聚类结果，返回 cluster_id 列表。"""
        conn = self._get_conn()
        ids = []
        for cluster in clusters:
            cursor = conn.execute(
                """INSERT INTO clusters (name, description, compile_run_id)
                   VALUES (?, ?, ?)""",
                (cluster["name"], cluster["description"], compile_run_id),
            )
            cid = cursor.lastrowid
            ids.append(cid)
            # 关联文档
            for doc_id in cluster.get("doc_ids", []):
                conn.execute(
                    "INSERT OR IGNORE INTO document_clusters (doc_id, cluster_id) VALUES (?, ?)",
                    (doc_id, cid),
                )
        conn.commit()
        return ids
```

**Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_registry.py -v`
Expected: 全部 PASS

**Step 5: Commit**

```bash
git add src/registry.py tests/test_registry.py
git commit -m "feat: add SQLite document registry with embedding cache"
```

---

## Phase 2: 文档接入层

### Task 4: Markdown 解析器

**Files:**
- Create: `src/ingester.py`
- Create: `tests/test_ingester.py`

**Step 1: 写测试**

Create `tests/test_ingester.py`:

```python
"""文档接入层测试。"""
import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("SILICONFLOW_API_KEY", "test-key")

from src.ingester import DocumentIngester, IngestResult


@pytest.fixture
def ingester():
    return DocumentIngester()


@pytest.fixture
def sample_md(tmp_path):
    """创建测试用 Markdown 文件。"""
    md_file = tmp_path / "test.md"
    md_file.write_text(
        "# 测试文档\n\n"
        "## 第一章 简介\n\n"
        "这是简介内容。\n\n"
        "## 第二章 详情\n\n"
        "### 2.1 子节\n\n"
        "详细内容。\n",
        encoding="utf-8",
    )
    return md_file


def test_ingest_markdown(ingester, sample_md):
    """Markdown 解析：提取标题层级和文本。"""
    result = ingester.ingest(sample_md)
    assert isinstance(result, IngestResult)
    assert "测试文档" in result.title
    assert len(result.headings) >= 2
    assert result.has_structure is True


def test_ingest_markdown_flat(tmp_path, ingester):
    """扁平 Markdown：无标题结构。"""
    flat_md = tmp_path / "flat.md"
    flat_md.write_text("这是一段普通文本，没有标题结构。\n", encoding="utf-8")
    result = ingester.ingest(flat_md)
    assert result.has_structure is False
    assert len(result.headings) == 0


def test_unsupported_format(tmp_path, ingester):
    """不支持的格式抛异常。"""
    bad_file = tmp_path / "test.xyz"
    bad_file.write_text("content", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持的格式"):
        ingester.ingest(bad_file)
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_ingester.py -v`
Expected: FAIL

**Step 3: 实现 ingester.py（Markdown 部分）**

Create `src/ingester.py`:

```python
"""文档接入层 — 统一解析 PDF / Markdown / DOCX / HTML 为 IngestResult。"""
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# 支持的格式映射
SUPPORTED_FORMATS = {
    ".pdf": "pdf",
    ".md": "md",
    ".markdown": "md",
    ".docx": "docx",
    ".doc": "docx",
    ".html": "html",
    ".htm": "html",
}


@dataclass
class Heading:
    level: int       # 1-6
    text: str


@dataclass
class IngestResult:
    text: str
    title: str
    headings: list[Heading] = field(default_factory=list)
    has_tables: bool = False
    has_lists: bool = False
    metadata: dict = field(default_factory=dict)


class DocumentIngester:
    """统一文档接入器。"""

    def ingest(self, file_path: Path) -> IngestResult:
        """解析文档，返回统一结构。"""
        file_path = Path(file_path)
        fmt = self._detect_format(file_path)
        parser = self._parsers.get(fmt)
        if parser is None:
            raise ValueError(f"不支持的格式: {file_path.suffix}")
        logger.info("解析文档: %s (格式: %s)", file_path.name, fmt)
        return parser(self, file_path)

    def _detect_format(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        fmt = SUPPORTED_FORMATS.get(suffix)
        if fmt is None:
            raise ValueError(f"不支持的格式: {suffix}")
        return fmt

    def _parse_markdown(self, file_path: Path) -> IngestResult:
        """解析 Markdown 文件。"""
        content = file_path.read_text(encoding="utf-8")
        headings: list[Heading] = []
        lines = content.split("\n")

        for line in lines:
            match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match:
                headings.append(Heading(
                    level=len(match.group(1)),
                    text=match.group(2).strip(),
                ))

        title = headings[0].text if headings else file_path.stem
        has_tables = "|" in content and "---" in content
        has_lists = bool(re.search(r"^\s*[-*+]\s", content, re.MULTILINE))

        return IngestResult(
            text=content,
            title=title,
            headings=headings,
            has_tables=has_tables,
            has_lists=has_lists,
            metadata={"format": "md", "line_count": len(lines)},
        )

    # 格式分派表
    _parsers = {
        "md": _parse_markdown,
    }
```

**Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_ingester.py -v`
Expected: 全部 PASS

**Step 5: Commit**

```bash
git add src/ingester.py tests/test_ingester.py
git commit -m "feat: add Markdown document ingester"
```

---

### Task 5: PDF 解析器

**Files:**
- Modify: `src/ingester.py`
- Modify: `tests/test_ingester.py`

**Step 1: 写测试**

在 `tests/test_ingester.py` 中追加：

```python
def test_ingest_pdf(tmp_path, ingester):
    """PDF 解析：提取文本和结构。"""
    # 创建一个简单的 PDF（使用 PyMuPDF）
    import fitz
    pdf_file = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "第一章 测试标题", fontsize=18)
    page.insert_text((72, 120), "这是正文内容。", fontsize=11)
    doc.save(str(pdf_file))
    doc.close()

    result = ingester.ingest(pdf_file)
    assert isinstance(result, IngestResult)
    assert "测试" in result.text or len(result.text) > 0
    assert result.metadata.get("format") == "pdf"
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_ingester.py::test_ingest_pdf -v`
Expected: FAIL — PDF 解析器未注册

**Step 3: 在 ingester.py 的 DocumentIngester 类中添加 PDF 解析**

在 `_parse_markdown` 方法后添加：

```python
    def _parse_pdf(self, file_path: Path) -> IngestResult:
        """解析 PDF 文件，使用 PyMuPDF。"""
        import fitz

        doc = fitz.open(str(file_path))
        text_parts: list[str] = []
        headings: list[Heading] = []

        for page_num, page in enumerate(doc, 1):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] != 0:  # 文本块
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        text_parts.append(text)
                        # 字号 > 14 视为标题
                        if span["size"] > 14:
                            headings.append(Heading(level=1, text=text))

        full_text = "\n".join(text_parts)
        title = headings[0].text if headings else file_path.stem

        return IngestResult(
            text=full_text,
            title=title,
            headings=headings,
            has_tables=False,
            has_lists=False,
            metadata={"format": "pdf", "page_count": len(doc)},
        )
```

在 `_parsers` 字典中添加：

```python
    _parsers = {
        "md": _parse_markdown,
        "pdf": _parse_pdf,
    }
```

**Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_ingester.py -v`
Expected: 全部 PASS

**Step 5: Commit**

```bash
git add src/ingester.py tests/test_ingester.py
git commit -m "feat: add PDF document ingester via PyMuPDF"
```

---

### Task 6: DOCX 解析器

**Files:**
- Modify: `src/ingester.py`
- Modify: `tests/test_ingester.py`

**Step 1: 写测试**

```python
def test_ingest_docx(tmp_path, ingester):
    """DOCX 解析：提取段落和标题样式。"""
    from docx import Document
    docx_file = tmp_path / "test.docx"
    doc = Document()
    doc.add_heading("测试文档标题", level=1)
    doc.add_paragraph("这是正文段落。")
    doc.add_heading("第一章", level=2)
    doc.add_paragraph("章节内容。")
    doc.save(str(docx_file))

    result = ingester.ingest(docx_file)
    assert isinstance(result, IngestResult)
    assert len(result.headings) >= 1
    assert result.has_structure is True
    assert result.metadata.get("format") == "docx"
```

**Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_ingester.py::test_ingest_docx -v`

**Step 3: 在 DocumentIngester 中添加 `_parse_docx`**

```python
    def _parse_docx(self, file_path: Path) -> IngestResult:
        """解析 DOCX 文件。"""
        from docx import Document

        doc = Document(str(file_path))
        text_parts: list[Heading] = []
        headings: list[Heading] = []
        has_tables = len(doc.tables) > 0

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            text_parts.append(text)
            if para.style.name.startswith("Heading"):
                try:
                    level = int(para.style.name.split()[-1])
                except (ValueError, IndexError):
                    level = 1
                headings.append(Heading(level=level, text=text))

        full_text = "\n".join(text_parts)
        title = headings[0].text if headings else file_path.stem

        return IngestResult(
            text=full_text,
            title=title,
            headings=headings,
            has_tables=has_tables,
            has_lists=False,
            metadata={"format": "docx", "paragraph_count": len(doc.paragraphs)},
        )
```

在 `_parsers` 中添加 `"docx": _parse_docx`。

**Step 4: 测试通过后 Commit**

```bash
git add src/ingester.py tests/test_ingester.py
git commit -m "feat: add DOCX document ingester via python-docx"
```

---

### Task 7: HTML 解析器

**Files:**
- Modify: `src/ingester.py`
- Modify: `tests/test_ingester.py`

**Step 1: 写测试**

```python
def test_ingest_html(tmp_path, ingester):
    """HTML 解析：提取标题和段落。"""
    html_file = tmp_path / "test.html"
    html_file.write_text(
        "<html><body>"
        "<h1>测试页面</h1>"
        "<p>这是正文。</p>"
        "<h2>子节</h2>"
        "<p>子节内容。</p>"
        "<ul><li>列表项1</li><li>列表项2</li></ul>"
        "</body></html>",
        encoding="utf-8",
    )

    result = ingester.ingest(html_file)
    assert isinstance(result, IngestResult)
    assert len(result.headings) == 2
    assert result.has_structure is True
    assert result.has_lists is True
    assert result.metadata.get("format") == "html"
```

**Step 2: 运行测试确认失败 → 实现 → 通过 → Commit**

添加 `_parse_html` 方法，用 BeautifulSoup4 解析 h1-h6、p、table、ul/ol。

在 `_parsers` 中添加 `"html": _parse_html`。

```bash
git add src/ingester.py tests/test_ingester.py
git commit -m "feat: add HTML document ingester via BeautifulSoup"
```

---

## Phase 3: 自动分类器

### Task 8: Token 计数 + 分类逻辑

**Files:**
- Create: `src/classifier.py`
- Create: `tests/test_classifier.py`

**Step 1: 写测试**

```python
"""自动分类器测试。"""
import os
from pathlib import Path

import pytest

os.environ.setdefault("SILICONFLOW_API_KEY", "test-key")

from src.classifier import DocumentClassifier
from src.ingester import IngestResult, Heading


@pytest.fixture
def classifier():
    return DocumentClassifier()


def test_classify_long_structured_doc(classifier):
    """有章节结构 + 超过阈值 → long。"""
    result = IngestResult(
        text="x" * 2000,
        title="长文档",
        headings=[
            Heading(level=1, text="第一章"),
            Heading(level=2, text="1.1 子节"),
            Heading(level=2, text="1.2 子节"),
        ],
        has_structure=True,
    )
    tier = classifier.classify(result)
    assert tier == "long"


def test_classify_short_flat_doc(classifier):
    """无结构 + 短文本 → short。"""
    result = IngestResult(
        text="这是一段短文本。",
        title="短文档",
        headings=[],
        has_structure=False,
    )
    tier = classifier.classify(result)
    assert tier == "short"


def test_classify_long_by_length_only(classifier):
    """无明确结构但超长 → long（长度权重 0.6 足够）。"""
    result = IngestResult(
        text="x" * 5000,
        title="超长无结构",
        headings=[Heading(level=1, text="唯一标题")],
        has_structure=False,
    )
    tier = classifier.classify(result)
    assert tier == "long"


def test_estimate_tokens_chinese(classifier):
    """中文 token 估算。"""
    count = classifier.estimate_tokens("这是一段中文文本。")
    assert count > 0
    # 大约 1.5 token/字
    assert 8 <= count <= 30


def test_estimate_tokens_english(classifier):
    """英文 token 估算。"""
    count = classifier.estimate_tokens("This is an English text for testing.")
    assert count > 0
```

**Step 2: 运行测试确认失败 → 实现 classifier.py → 通过 → Commit**

Create `src/classifier.py`:

```python
"""自动长短文档分类器 — 基于 token 数和结构特征加权判断。"""
import logging
import re

from src.ingester import IngestResult
import src.config as cfg

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """根据结构和长度自动判断文档 tier。"""

    def classify(self, result: IngestResult) -> str:
        """返回 'short' 或 'long'。"""
        structure_score = self._structure_score(result)
        length_score = self._length_score(result)

        final = (
            structure_score * cfg.CLASSIFIER_STRUCTURE_WEIGHT
            + length_score * cfg.CLASSIFIER_LENGTH_WEIGHT
        )

        tier = "long" if final > 0.5 else "short"
        token_count = self.estimate_tokens(result.text)
        logger.info(
            "分类结果: tier=%s, structure=%.2f, length=%.2f, final=%.2f, tokens=%d",
            tier, structure_score, length_score, final, token_count,
        )
        return tier

    def estimate_tokens(self, text: str) -> int:
        """估算 token 数（字符模式，零依赖）。"""
        if not text:
            return 0
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        non_chinese = re.sub(r"[\u4e00-\u9fff]", " ", text)
        english_words = len(non_chinese.split())
        return int(chinese_chars * 1.5 + english_words * 1.3)

    def _structure_score(self, result: IngestResult) -> float:
        """结构评分 0-1。检测标题层级深度和多样性。"""
        if not result.headings:
            return 0.0
        levels = {h.level for h in result.headings}
        depth = max(levels) - min(levels) + 1 if levels else 0
        heading_count = len(result.headings)

        # 层级深度 ≥ 2 或标题数 ≥ 3 → 高分
        score = min(1.0, (depth * 0.3 + heading_count * 0.1))
        if result.has_tables:
            score = min(1.0, score + 0.1)
        return score

    def _length_score(self, result: IngestResult) -> float:
        """长度评分 0-1，基于 token 阈值。"""
        tokens = self.estimate_tokens(result.text)
        if tokens == 0:
            return 0.0
        # 超过阈值 → 1.0，否则按比例
        ratio = tokens / cfg.CLASSIFIER_TOKEN_THRESHOLD
        return min(1.0, ratio)
```

```bash
git add src/classifier.py tests/test_classifier.py
git commit -m "feat: add document auto-classifier with structure+length scoring"
```

---

## Phase 4: 增量编译引擎

### Task 9: 编译引擎核心

**Files:**
- Create: `src/compiler.py`
- Create: `tests/test_compiler.py`

这是最核心的模块，整合 registry + ingester + classifier + embedding + 聚类。实现步骤：

**Step 1: 写测试** — 测试增量编译逻辑（变更检测、跳过、重编译）

**Step 2: 实现 compiler.py**

关键方法：
- `scan_data_dir()` — 扫描 data/ 目录，与 SQLite hash 对比
- `process_document()` — 单文档 Ingest → Classify → Embed 流水线
- `compile()` — 完整编译流程：扫描 → 处理 → 聚类 → 物化
- `_run_embedding()` — 调 API 获取向量，缓存到 SQLite
- `_run_clustering()` — KMeans 聚类 + LLM 命名（复用现有 `_name_cluster` 逻辑）

**Step 3: 测试通过 → Commit**

```bash
git add src/compiler.py tests/test_compiler.py
git commit -m "feat: add incremental compilation engine"
```

---

### Task 10: 原子物化模块

**Files:**
- Create: `src/materializer.py`
- Create: `tests/test_materializer.py`

**Step 1: 写测试** — 验证 tmp → bak → rename 原子替换，以及 INDEX.md 格式（含来源信息）

**Step 2: 实现 materializer.py**

```python
class Materializer:
    def materialize(self, clusters, documents, compile_run_id):
        # 1. 写 compiled_library.tmp/
        # 2. 删除旧的 compiled_library.bak/
        # 3. rename compiled_library/ → compiled_library.bak/
        # 4. rename compiled_library.tmp/ → compiled_library/
```

INDEX.md 格式：
```markdown
- [短文档] doc_id: 标题 (来源: path/to/file.pdf)
- [长文档] pageindex-uuid: 标题 (来源: path/to/file.pdf, PageIndex索引)
```

**Step 3: 测试通过 → Commit**

```bash
git add src/materializer.py tests/test_materializer.py
git commit -m "feat: add atomic materialization with source info"
```

---

## Phase 5: 触发模式

### Task 11: CLI 触发器 + __main__.py

**Files:**
- Create: `src/triggers/__init__.py`
- Create: `src/triggers/cli.py`
- Create: `src/__main__.py`

```bash
# 用法
python -m src compile              # 增量
python -m src compile --force       # 全量
python -m src status                # 查看状态
```

**Step 1: 写测试** — 验证 CLI 参数解析和 compiler 调用

**Step 2: 实现** — argparse + 调用 `compiler.compile(trigger_type="manual")`

**Step 3: 测试通过 → Commit**

```bash
git add src/__main__.py src/triggers/__init__.py src/triggers/cli.py tests/test_cli.py
git commit -m "feat: add CLI trigger for manual compilation"
```

---

### Task 12: 文件监控触发器

**Files:**
- Create: `src/triggers/watcher.py`
- Create: `tests/test_watcher.py`

watchdog 库监听 data/ 目录变化，500ms 防抖，触发增量编译。

**测试**: 模拟文件创建/修改事件，验证防抖逻辑。

```bash
git add src/triggers/watcher.py tests/test_watcher.py
git commit -m "feat: add file watcher trigger with debouncing"
```

---

### Task 13: 定时调度触发器

**Files:**
- Create: `src/triggers/scheduler.py`
- Create: `tests/test_scheduler.py`

APScheduler cron 触发增量编译。

```bash
git add src/triggers/scheduler.py tests/test_scheduler.py
git commit -m "feat: add scheduled trigger via APScheduler"
```

---

### Task 14: FastAPI 触发器

**Files:**
- Create: `src/triggers/api_server.py`
- Create: `tests/test_api_server.py`

端点：
- `POST /api/compile` → 触发编译
- `GET /api/status` → 注册表状态
- `GET /api/documents` → 文档列表
- `POST /api/documents/upload` → 上传并编译
- `GET /api/health` → 健康检查

**测试**: 使用 FastAPI TestClient 验证端点。

```bash
git add src/triggers/api_server.py tests/test_api_server.py
git commit -m "feat: add FastAPI trigger with upload and health endpoints"
```

---

## Phase 6: Agent 兼容性改造

### Task 15: 更新 main_agent.py

**Files:**
- Modify: `src/main_agent.py`

改动点：
1. 初始化 SQLite 连接（`KnowledgeRegistry`）
2. `get_short_document` 改为从 SQLite 查询（保留 JSON 降级）
3. 新增 `get_document_source` 工具
4. 保留现有 4 个工具接口不变

**验证**: 运行 `python -m src.main_agent`，确认 Agent 仍能正常工作。

```bash
git add src/main_agent.py
git commit -m "feat: add document source tool and SQLite-based short doc lookup to agent"
```

---

### Task 16: 端到端验证

**Files:**
- 无新文件

**Step 1: 全流程验证**

```bash
# 1. 清理旧产物
rm -rf compiled_library/ knowledge.db

# 2. 放入真实文档（用现有 data/ 中的文件）
# 确保 data/short_docs.json 和 data/travel_policy_2026.md 存在

# 3. 运行编译
python -m src compile

# 4. 启动 Agent
python -m src.main_agent
# 测试查询: "如何做报销申请？"
# 测试查询: "VPN 怎么连？"
```

**Step 2: 验证兼容性**

- SKILL.md 格式与旧版一致
- Agent routing 正常工作
- PageIndex 长文档检索正常
- 来源信息正确显示

---

## 依赖关系

```
Task 1 (deps) → Task 2 (config) → Task 3 (registry)
                                    ↓
Task 4-7 (ingester: md/pdf/docx/html) → Task 8 (classifier)
                                           ↓
                                       Task 9 (compiler) → Task 10 (materializer)
                                                              ↓
                                              Task 11-14 (triggers: CLI/watch/sched/api)
                                                              ↓
                                              Task 15 (agent update) → Task 16 (E2E)
```

Task 4-7 可并行开发。Task 11-14 可并行开发。
