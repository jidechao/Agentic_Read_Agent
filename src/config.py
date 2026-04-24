"""统一配置 — 所有可调参数集中管理，启动时 fail-fast 检查环境变量。"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── PageIndex 路径注册（必须在 import pageindex 之前）──────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PAGEINDEX_SRC = _PROJECT_ROOT / "PageIndex"
if str(_PAGEINDEX_SRC) not in sys.path:
    sys.path.insert(0, str(_PAGEINDEX_SRC))

# ── API 配置 ──────────────────────────────────────────────────────────
SILICONFLOW_API_KEY: str = os.environ.get("SILICONFLOW_API_KEY", "")
if not SILICONFLOW_API_KEY:
    raise RuntimeError("环境变量 SILICONFLOW_API_KEY 未设置，请在 .env 中配置")

SILICONFLOW_BASE_URL: str = "https://api.siliconflow.cn/v1"

# ── 模型配置（可热换）──────────────────────────────────────────────────
EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-4B"
EMBEDDING_DIMENSION: int = 1536
AGENT_MODEL: str = "Qwen/Qwen3-32B"
CLUSTER_NAMING_MODEL: str = "Qwen/Qwen3-8B"
PAGEINDEX_MODEL: str = "Qwen/Qwen3-8B"

# ── 路径配置 ──────────────────────────────────────────────────────────
PROJECT_ROOT: Path = _PROJECT_ROOT
DATA_DIR: Path = PROJECT_ROOT / "data"
COMPILED_DIR: Path = PROJECT_ROOT / "compiled_library"
PAGEINDEX_WORKSPACE: str = str(COMPILED_DIR / "pageindex_cache")
SHORT_DOCS_DB: Path = COMPILED_DIR / "short_docs_db.json"
SKILL_MD: Path = COMPILED_DIR / "SKILL.md"

# ── 聚类配置 ──────────────────────────────────────────────────────────
CLUSTER_K: int = 2
LONG_DOC_SUMMARY_LENGTH: int = 500  # 长文档聚类摘要截取字数
