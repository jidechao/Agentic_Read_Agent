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
AGENT_MODEL: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
CLUSTER_NAMING_MODEL: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PAGEINDEX_MODEL: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# ── 路径配置 ──────────────────────────────────────────────────────────
PROJECT_ROOT: Path = _PROJECT_ROOT
DATA_DIR: Path = PROJECT_ROOT / "data"
COMPILED_DIR: Path = PROJECT_ROOT / "compiled_library"
PAGEINDEX_WORKSPACE: str = str(COMPILED_DIR / "pageindex_cache")
SHORT_DOCS_DB: Path = COMPILED_DIR / "short_docs_db.json"
SKILL_MD: Path = COMPILED_DIR / "SKILL.md"

# ── 聚类配置 ──────────────────────────────────────────────────────────
# 兼容旧版 offline_compiler.py；新增量编译管线不再使用固定 K。
CLUSTER_K: int = int(os.environ.get("CLUSTER_K", "0"))  # 0 = auto (sqrt(n/2))
LONG_DOC_SUMMARY_LENGTH: int = 500  # 长文档聚类摘要截取字数
CLUSTER_COSINE_THRESHOLD: float = float(os.environ.get("CLUSTER_COSINE_THRESHOLD", "0.30"))
INCREMENTAL_ASSIGN_THRESHOLD: float = float(os.environ.get("INCREMENTAL_ASSIGN_THRESHOLD", "0.55"))
PENDING_REVIEW_REBALANCE_RATIO: float = float(os.environ.get("PENDING_REVIEW_REBALANCE_RATIO", "0.10"))
PENDING_REVIEW_REBALANCE_ABSOLUTE: int = int(os.environ.get("PENDING_REVIEW_REBALANCE_ABSOLUTE", "30"))
LLM_SUMMARY_MAX_CHARS: int = int(os.environ.get("LLM_SUMMARY_MAX_CHARS", "80"))
TOPIC_SUMMARY_MODEL: str = os.environ.get("TOPIC_SUMMARY_MODEL", CLUSTER_NAMING_MODEL)
CATEGORY_REUSE_THRESHOLD: float = float(os.environ.get("CATEGORY_REUSE_THRESHOLD", "0.65"))
PENDING_REVIEW_REBALANCE_MIN: int = int(os.environ.get("PENDING_REVIEW_REBALANCE_MIN", "5"))

# ── 自动分类器配置 ──────────────────────────────────────────────────────
CLASSIFIER_TOKEN_THRESHOLD: int = int(os.environ.get("CLASSIFIER_TOKEN_THRESHOLD", "1000"))
CLASSIFIER_STRUCTURE_WEIGHT: float = float(os.environ.get("CLASSIFIER_STRUCTURE_WEIGHT", "0.4"))
CLASSIFIER_LENGTH_WEIGHT: float = float(os.environ.get("CLASSIFIER_LENGTH_WEIGHT", "0.6"))
CLASSIFIER_USE_EXACT_TOKENS: bool = os.environ.get("CLASSIFIER_USE_EXACT_TOKENS", "false").lower() == "true"

# ── 注册表路径 ──────────────────────────────────────────────────────────
KNOWLEDGE_DB: Path = PROJECT_ROOT / "knowledge.db"
