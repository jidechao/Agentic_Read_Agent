# Two-Tier Active Navigation RAG Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a two-tier active navigation RAG system with SiliconFlow LLM, PageIndex long-doc indexing, KMeans clustering, and an openai-agents interactive navigator.

**Architecture:** Offline compiler generates a two-tier directory (SKILL.md + INDEX.md files) from raw documents using PageIndex + embedding clustering. Online agent uses 4 tools to navigate the directory and retrieve content. All LLM calls go through SiliconFlow.

**Tech Stack:** Python 3.11, openai-agents SDK, SiliconFlow API (Qwen3), pageindex (local), scikit-learn, python-dotenv

---

## Task 1: Environment Setup

**Files:**
- Create: `.env`
- Create: `src/__init__.py` (empty)
- Modify: `.venv` (install deps)

**Step 1: Create .env**

```bash
cat > /Volumes/MyWork/project/agentic_read_agent/.env << 'EOF'
SILICONFLOW_API_KEY=your_key_here
EOF
```

**Step 2: Create src directory and install dependencies**

```bash
cd /Volumes/MyWork/project/agentic_read_agent
mkdir -p src data compiled_library
touch src/__init__.py
source .venv/bin/activate
pip install openai openai-agents scikit-learn python-dotenv numpy requests
pip install -e ./PageIndex
```

**Step 3: Verify installation**

```bash
python -c "from agents import Agent, Runner, function_tool; print('openai-agents OK')"
python -c "from pageindex import PageIndexClient; print('pageindex OK')"
python -c "from sklearn.cluster import KMeans; print('sklearn OK')"
python -c "from openai import OpenAI; print('openai OK')"
```

Expected: All 4 print statements succeed.

**Step 4: Commit**

```bash
git init
git add .env.example src/__init__.py
git commit -m "chore: initialize project structure and dependencies"
```

Note: Do NOT commit `.env` with real keys. Create `.gitignore` with `.env` entry.

---

## Task 2: Configuration Module

**Files:**
- Create: `src/config.py`

**Step 1: Write src/config.py**

```python
"""统一配置 — 所有可调参数集中管理，启动时 fail-fast 检查环境变量。"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

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
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
COMPILED_DIR: Path = PROJECT_ROOT / "compiled_library"
PAGEINDEX_WORKSPACE: str = str(COMPILED_DIR / "pageindex_cache")
SHORT_DOCS_DB: Path = COMPILED_DIR / "short_docs_db.json"
SKILL_MD: Path = COMPILED_DIR / "SKILL.md"

# ── 聚类配置 ──────────────────────────────────────────────────────────
CLUSTER_K: int = 2
LONG_DOC_SUMMARY_LENGTH: int = 500  # 长文档聚类摘要截取字数
```

**Step 2: Verify config loads**

```bash
SILICONFLOW_API_KEY=test python -c "import sys; sys.path.insert(0,'src'); import config; print(f'BASE_URL={config.SILICONFLOW_BASE_URL}, EMBEDDING={config.EMBEDDING_MODEL}')"
```

Expected: Prints config values without error.

**Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: add centralized configuration module"
```

---

## Task 3: Data Generator

**Files:**
- Create: `src/data_generator.py`

**Step 1: Write src/data_generator.py**

```python
"""生成测试数据：4 篇短文档 + 1 篇长文档。
短文档存为 data/short_docs.json，长文档存为 data/travel_policy_2026.md。"""
import json
from pathlib import Path
import src.config as cfg


def generate_short_docs() -> list[dict]:
    """生成 4 篇短文档，涵盖 IT 和 HR 政策。"""
    return [
        {
            "doc_id": "doc_001",
            "title": "IT 设备申请流程",
            "content": (
                "所有员工如需申请新的办公设备（包括笔记本电脑、显示器、键盘鼠标等），"
                "须通过 OA 系统提交「IT 设备申请单」。申请需经直属上级审批后，"
                "由 IT 部门统一采购。标准配置笔记本电脑的审批周期为 3-5 个工作日。"
                "紧急需求可联系 IT 服务热线 8888 加急处理。设备到货后需在 IT 资产管理系统中登记，"
                "员工离职时须归还所有 IT 资产。"
            ),
        },
        {
            "doc_id": "doc_002",
            "title": "VPN 远程接入指南",
            "content": (
                "公司提供 VPN 服务供员工远程办公使用。首次使用请按以下步骤操作：\n"
                "1. 从内网下载 VPN 客户端安装包（支持 Windows/macOS）\n"
                "2. 使用工号和域密码登录\n"
                "3. 选择「默认」连接配置\n"
                "4. 连接成功后可访问内网资源\n"
                "注意：VPN 连接后不可同时使用公司 Wi-Fi，否则会造成网络冲突。"
                "如遇连接问题请联系 IT 服务台。每日 VPN 使用时长限制为 10 小时。"
            ),
        },
        {
            "doc_id": "doc_003",
            "title": "出差住宿标准",
            "content": (
                "根据公司差旅管理制度，员工出差住宿标准如下：\n"
                "- 一线城市（北京、上海、广州、深圳）：标准间不超过 600 元/晚\n"
                "- 二线城市：标准间不超过 400 元/晚\n"
                "- 三线及以下城市：标准间不超过 300 元/晚\n"
                "- 管理层（总监及以上）可在上述标准基础上上浮 50%\n"
                "住宿发票须注明入住人姓名和入住日期，退房时主动索取。"
                "超标部分需提前申请特殊审批，未经批准的超标费用不予报销。"
            ),
        },
        {
            "doc_id": "doc_004",
            "title": "年度体检安排",
            "content": (
                "公司每年为全体在职员工安排一次免费健康体检。体检安排如下：\n"
                "- 体检时间：每年 4 月至 6 月\n"
                "- 体检机构：美年大健康（全国连锁）\n"
                "- 预约方式：通过 HR 系统在线预约\n"
                "- 体检前一天 22:00 后禁食禁水\n"
                "- 体检报告于检查后 7 个工作日内在 HR 系统查看\n"
                "员工可根据自身需求自费加选体检项目。未在规定时间内完成体检视为放弃该年度权益。"
            ),
        },
    ]


LONG_DOC_CONTENT = """# 2026年公司差旅报销政策

## 第一章 总则

### 1.1 政策目的

为规范公司员工差旅报销行为，加强费用管控，根据国家相关法律法规和公司财务管理制度，特制定本政策。本政策适用于公司全体全职员工、兼职员工及经批准的外部合作人员。

### 1.2 适用范围

本政策适用于员工因公出差产生的交通、住宿、餐饮及其他相关费用的报销。出差须事先获得直属上级和部门负责人的书面批准（通过 OA 系统提交出差申请）。

### 1.3 报销原则

- 实报实销原则：凭有效票据报销实际发生的合理费用
- 标准控制原则：各类费用不超过规定标准
- 时效原则：出差返回后 15 个工作日内完成报销申请

## 第二章 交通费用

### 2.1 交通工具标准

| 职级 | 飞机 | 高铁/动车 | 其他 |
|------|------|-----------|------|
| 总监及以上 | 经济舱 | 一等座 | 实报实销 |
| 经理级 | 经济舱 | 二等座 | 实报实销 |
| 普通员工 | 提前 7 天预订经济舱特价票 | 二等座 | 公交/地铁/打车 |

### 2.2 市内交通

出差目的地的市内交通费用（出租车、网约车、公共交通）按实际发生额报销，单次打车不超过 100 元。如需租车，须提前申请并获得行政部门批准。

## 第三章 住宿费用

### 3.1 住宿标准

- 一线城市（北京、上海、广州、深圳）：不超过 600 元/晚
- 二线城市：不超过 400 元/晚
- 三线及以下城市：不超过 300 元/晚
- 管理层可在上述标准基础上上浮 50%

### 3.2 住宿注意事项

住宿须选择正规酒店并索取增值税专用发票。发票须注明入住人姓名、入住日期和退房日期。连续住宿超过 7 天的，建议与酒店协商长住价格。

## 第四章 餐饮补贴

### 4.1 餐补标准

出差期间的餐饮补贴按天计算：
- 一线城市：120 元/天
- 二线城市：100 元/天
- 三线及以下城市：80 元/天

### 4.2 宴请费用

因公宴请客户须提前申请，审批通过后按实际发生额报销，单次不超过 2000 元。宴请须两人以上参加，并提交参与人员名单和宴请事由说明。

## 第五章 票据遗失处理

### 5.1 交通票据遗失

如交通票据遗失，按以下流程处理：
1. 联系出票方补打或获取电子票据
2. 如无法补打，提交书面情况说明（含出差事由、行程信息、金额）
3. 书面说明须经直属上级签字确认
4. 财务部审核后按票面金额的 80% 报销

### 5.2 住宿票据遗失

住宿发票遗失的处理：
1. 联系酒店补开发票（酒店有义务在当年内补开）
2. 如跨年无法补开，提交酒店入住确认单和付款截图
3. 经部门负责人和财务总监双签确认后方可报销

### 5.3 餐饮票据遗失

餐饮票据遗失原则上不予报销。特殊情况下（如集体出差发票由一人集中开具后遗失），提交书面说明并经部门负责人签字后，按餐补标准报销。

## 第六章 报销流程

### 6.1 报销申请

出差返回后 15 个工作日内，通过 OA 系统提交报销申请：
1. 填写报销单（出差日期、目的地、事由）
2. 上传所有票据扫描件/照片
3. 选择费用归属项目
4. 提交审批

### 6.2 审批流程

- 5000 元以下：直属上级 → 财务审核
- 5000-20000 元：直属上级 → 部门负责人 → 财务审核
- 20000 元以上：直属上级 → 部门负责人 → 财务总监 → CFO 审批

### 6.3 支付时间

审批通过后 5 个工作日内，财务部将报销款项转入员工工资账户。
"""


def generate_long_doc() -> None:
    """生成长文档 travel_policy_2026.md。"""
    output_path = cfg.DATA_DIR / "travel_policy_2026.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(LONG_DOC_CONTENT, encoding="utf-8")
    print(f"已生成长文档: {output_path}")


def generate_all() -> dict:
    """生成所有测试数据，返回短文档 dict。"""
    # 短文档
    short_docs = generate_short_docs()
    short_path = cfg.DATA_DIR / "short_docs.json"
    short_path.parent.mkdir(parents=True, exist_ok=True)
    short_path.write_text(
        json.dumps(short_docs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已生成短文档: {short_path} ({len(short_docs)} 篇)")

    # 长文档
    generate_long_doc()

    return {"short_docs": short_docs}


if __name__ == "__main__":
    generate_all()
    print("数据生成完毕。")
```

**Step 2: Run and verify**

```bash
cd /Volumes/MyWork/project/agentic_read_agent
python -m src.data_generator
```

Expected:
```
已生成短文档: /Volumes/MyWork/project/agentic_read_agent/data/short_docs.json (4 篇)
已生成长文档: /Volumes/MyWork/project/agentic_read_agent/data/travel_policy_2026.md
数据生成完毕。
```

**Step 3: Verify output files**

```bash
python -c "import json; d=json.load(open('data/short_docs.json')); print(f'{len(d)} docs, IDs: {[x[\"doc_id\"] for x in d]}')"
head -3 data/travel_policy_2026.md
```

Expected: `4 docs, IDs: ['doc_001', 'doc_002', 'doc_003', 'doc_004']` and `# 2026年公司差旅报销政策`

**Step 4: Commit**

```bash
git add src/data_generator.py data/
git commit -m "feat: add data generator with 4 short docs + 1 long doc"
```

---

## Task 4: Offline Compiler — Part 1 (PageIndex + Embedding)

**Files:**
- Create: `src/offline_compiler.py`

**Step 1: Write the full offline_compiler.py**

```python
"""双层离线编译引擎。

Tier 2 长文档建树 (PageIndex) + Tier 1 Embedding 聚类 + 簇命名 → 物化输出。

参考: agentic_vectorless_rag_demo.py 中已验证的 PageIndex 初始化和调用模式。
"""
import os
import json
import logging
from pathlib import Path

import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans

import src.config as cfg

logger = logging.getLogger(__name__)

# ── SiliconFlow 客户端（Embedding + Chat）──────────────────────────────
_sf_client = OpenAI(
    api_key=cfg.SILICONFLOW_API_KEY,
    base_url=cfg.SILICONFLOW_BASE_URL,
)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """调用 SiliconFlow Qwen3-Embedding-4B 对文本列表进行向量化。"""
    logger.info("正在向量化 %d 条文本...", len(texts))
    response = _sf_client.embeddings.create(
        model=cfg.EMBEDDING_MODEL,
        input=texts,
        dimensions=cfg.EMBEDDING_DIMENSION,
    )
    vectors = [item.embedding for item in response.data]
    logger.info("向量化完成，维度: %d", len(vectors[0]))
    return vectors


def init_pageindex_client():
    """初始化 PageIndexClient，通过环境变量传导 SiliconFlow 配置。

    PageIndex 内部使用 LiteLLM，读取 OPENAI_API_KEY 和 OPENAI_BASE_URL。
    """
    os.environ["OPENAI_API_KEY"] = cfg.SILICONFLOW_API_KEY
    os.environ["OPENAI_BASE_URL"] = cfg.SILICONFLOW_BASE_URL

    from pageindex import PageIndexClient

    client = PageIndexClient(
        workspace=cfg.PAGEINDEX_WORKSPACE,
        model=cfg.PAGEINDEX_MODEL,
    )
    logger.info("PageIndexClient 初始化完成，workspace: %s", cfg.PAGEINDEX_WORKSPACE)
    return client


def cluster_and_name(
    vectors: list[list[float]],
    doc_infos: list[dict],
    k: int = cfg.CLUSTER_K,
) -> list[dict]:
    """KMeans 聚类 + LLM 簇命名。

    doc_infos: [{"doc_id": str, "title": str, "type": "short"|"long"}, ...]
    返回: [{"name": str, "description": str, "doc_indices": [int, ...]}, ...]
    """
    logger.info("开始聚类 (K=%d)...", k)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)
    logger.info("聚类完成，标签分布: %s", dict(zip(*np.unique(labels, return_counts=True))))

    clusters = []
    for cluster_id in range(k):
        indices = [i for i, label in enumerate(labels) if label == cluster_id]
        titles = [doc_infos[i]["title"] for i in indices]
        logger.info("簇 %d 的文档: %s", cluster_id, titles)

        # 调 LLM 给簇命名
        name, description = _name_cluster(titles)
        clusters.append({
            "name": name,
            "description": description,
            "doc_indices": indices,
        })
        logger.info("簇 %d 命名为: %s — %s", cluster_id, name, description)

    return clusters


def _name_cluster(doc_titles: list[str]) -> tuple[str, str]:
    """调用 SiliconFlow LLM 为文档簇起名。"""
    prompt = (
        f"以下是同一分类的企业知识文档标题列表：\n"
        + "\n".join(f"- {t}" for t in doc_titles)
        + "\n\n请用一个简短的英文短语作为分类目录名（小写，用连字符连接，如 it-hr），"
        "再写一句中文描述该分类。\n"
        '请严格按以下 JSON 格式返回：{"name": "xxx", "description": "xxx"}'
    )
    try:
        response = _sf_client.chat.completions.create(
            model=cfg.CLUSTER_NAMING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        text = response.choices[0].message.content.strip()
        # 尝试解析 JSON（处理可能的 markdown 代码块包裹）
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
        return result["name"], result["description"]
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("簇命名 LLM 返回解析失败: %s，使用默认名称", e)
        return f"cluster-default", "默认分类"


def materialize(
    clusters: list[dict],
    doc_infos: list[dict],
    short_docs: list[dict],
    pageindex_doc_id: str | None,
) -> None:
    """将编译结果物化到 compiled_library/ 目录。"""
    cfg.COMPILED_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. short_docs_db.json ──────────────────────────────────────────
    db = {doc["doc_id"]: doc for doc in short_docs}
    cfg.SHORT_DOCS_DB.write_text(
        json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("已写入: %s (%d 篇)", cfg.SHORT_DOCS_DB, len(db))

    # ── 2. 各子目录 INDEX.md ────────────────────────────────────────────
    for cluster in clusters:
        dir_name = cluster["name"]
        cluster_dir = cfg.COMPILED_DIR / dir_name
        cluster_dir.mkdir(parents=True, exist_ok=True)

        lines = [f"# {dir_name}", ""]
        for idx in cluster["doc_indices"]:
            info = doc_infos[idx]
            if info["type"] == "short":
                lines.append(f"- [短文档] {info['doc_id']}: {info['title']}")
            else:
                # 长文档用 pageindex 的 doc_id
                pid = pageindex_doc_id or info["doc_id"]
                lines.append(f"- [长文档] {pid}: {info['title']}")

        index_path = cluster_dir / "INDEX.md"
        index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("已写入: %s", index_path)

    # ── 3. SKILL.md 总目录 ─────────────────────────────────────────────
    skill_lines = ["# 知识库总目录", ""]
    for cluster in clusters:
        skill_lines.append(f"## [{cluster['name']}]({cluster['name']}/INDEX.md)")
        skill_lines.append(cluster["description"])
        skill_lines.append("")
    cfg.SKILL_MD.write_text("\n".join(skill_lines), encoding="utf-8")
    logger.info("已写入: %s", cfg.SKILL_MD)


def compile_library() -> None:
    """执行完整的离线编译流程。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # ── 1. 读取数据 ────────────────────────────────────────────────────
    short_docs = json.loads(
        (cfg.DATA_DIR / "short_docs.json").read_text(encoding="utf-8")
    )
    long_doc_path = str((cfg.DATA_DIR / "travel_policy_2026.md").resolve())
    logger.info("已读取 %d 篇短文档 + 1 篇长文档", len(short_docs))

    # ── 2. Tier 2: 长文档建树 ──────────────────────────────────────────
    pi_client = init_pageindex_client()

    # 检查是否已索引过（避免重复建树消耗 API 额度）
    pageindex_doc_id = None
    for did, doc in pi_client.documents.items():
        if doc.get("doc_name") == "travel_policy_2026.md":
            pageindex_doc_id = did
            logger.info("发现已缓存的 PageIndex 文档: %s", did)
            break

    if not pageindex_doc_id:
        logger.info("开始索引长文档...")
        pageindex_doc_id = pi_client.index(long_doc_path, mode="md")
        logger.info("索引完成，doc_id: %s", pageindex_doc_id)

    # 提取前 500 字作为聚类摘要
    long_doc_content = Path(long_doc_path).read_text(encoding="utf-8")
    long_summary = long_doc_content[:cfg.LONG_DOC_SUMMARY_LENGTH]

    # ── 3. 构建文档信息表 ──────────────────────────────────────────────
    doc_infos = []
    texts_for_embedding = []
    for doc in short_docs:
        doc_infos.append({
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "type": "short",
        })
        texts_for_embedding.append(doc["content"])

    doc_infos.append({
        "doc_id": pageindex_doc_id,
        "title": "2026年公司差旅报销政策",
        "type": "long",
    })
    texts_for_embedding.append(long_summary)

    # ── 4. 向量化 + 聚类 ───────────────────────────────────────────────
    vectors = get_embeddings(texts_for_embedding)
    clusters = cluster_and_name(vectors, doc_infos)

    # ── 5. 物化输出 ────────────────────────────────────────────────────
    materialize(clusters, doc_infos, short_docs, pageindex_doc_id)

    print("\n编译完成！目录结构：")
    print(f"  {cfg.SKILL_MD}")
    for cluster in clusters:
        print(f"  {cfg.COMPILED_DIR / cluster['name'] / 'INDEX.md'}")
    print(f"  {cfg.SHORT_DOCS_DB}")


if __name__ == "__main__":
    compile_library()
```

**Step 2: Run the full compile**

```bash
cd /Volumes/MyWork/project/agentic_read_agent
python -m src.offline_compiler
```

Expected:
```
INFO: 正在向量化 5 条文本...
INFO: 向量化完成，维度: 1536
INFO: 开始聚类 (K=2)...
INFO: 聚类完成...
INFO: 已写入: compiled_library/SKILL.md
INFO: 已写入: compiled_library/xxx/INDEX.md
编译完成！
```

**Step 3: Verify compiled output**

```bash
cat compiled_library/SKILL.md
cat compiled_library/*/INDEX.md
python -c "import json; db=json.load(open('compiled_library/short_docs_db.json')); print(f'{len(db)} docs: {list(db.keys())}')"
```

Expected: SKILL.md has cluster links, INDEX.md files have `[短文档]` and `[长文档]` entries, short_docs_db.json has 4 entries.

**Step 4: Commit**

```bash
git add src/offline_compiler.py compiled_library/
git commit -m "feat: add offline compiler with PageIndex + embedding clustering"
```

---

## Task 5: Main Agent

**Files:**
- Create: `src/main_agent.py`

**Step 1: Write src/main_agent.py**

```python
"""在线导航 Agent — 使用 openai-agents SDK 的双层知识库交互程序。

参考 agentic_vectorless_rag_demo.py 中已验证的：
- PageIndexClient 初始化和调用模式
- Runner.run_streamed 事件处理模式
- tool_call_item / tool_call_output_item / ResponseTextDeltaEvent 处理

关键：SiliconFlow 不支持 Responses API，需使用 OpenAIChatCompletionsModel。
"""
import os
import sys
import json
import asyncio
import logging
from pathlib import Path

from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents import (
    Agent,
    Runner,
    function_tool,
    set_tracing_disabled,
    set_default_openai_client,
)
from agents.models import OpenAIChatCompletionsModel

import src.config as cfg

logger = logging.getLogger(__name__)

# ── 全局初始化 ───────────────────────────────────────────────────────────

# 1) SiliconFlow 客户端（openai-agents SDK 用）
sf_async_client = AsyncOpenAI(
    api_key=cfg.SILICONFLOW_API_KEY,
    base_url=cfg.SILICONFLOW_BASE_URL,
)
set_default_openai_client(sf_async_client)
set_tracing_disabled(True)

# 2) PageIndexClient（环境变量传导 SiliconFlow）
os.environ["OPENAI_API_KEY"] = cfg.SILICONFLOW_API_KEY
os.environ["OPENAI_BASE_URL"] = cfg.SILICONFLOW_BASE_URL

from pageindex import PageIndexClient

pageindex_client = PageIndexClient(
    workspace=cfg.PAGEINDEX_WORKSPACE,
    model=cfg.PAGEINDEX_MODEL,
)

# 3) 加载短文档库
_short_docs_db: dict = {}
if cfg.SHORT_DOCS_DB.exists():
    _short_docs_db = json.loads(cfg.SHORT_DOCS_DB.read_text(encoding="utf-8"))


# ── 工具定义 ────────────────────────────────────────────────────────────

@function_tool
def read_library_directory(path: str) -> str:
    """读取 compiled_library 目录下的 .md 文件内容。
    path: 相对路径，如 'SKILL.md' 或 'finance-travel/INDEX.md'。
    """
    full_path = cfg.COMPILED_DIR / path
    if not full_path.exists():
        return f"错误：文件 {path} 不存在"
    try:
        content = full_path.read_text(encoding="utf-8")
        return content
    except Exception as e:
        return f"读取失败: {e}"


@function_tool
def get_short_document(doc_id: str) -> str:
    """从短文档库获取指定文档的完整内容。
    doc_id: 文档ID，如 'doc_001'。
    """
    doc = _short_docs_db.get(doc_id)
    if not doc:
        available = ", ".join(_short_docs_db.keys()) if _short_docs_db else "无"
        return f"错误：未找到文档 {doc_id}。可用文档: {available}"
    return json.dumps(doc, ensure_ascii=False)


@function_tool
def view_long_document_toc(doc_id: str) -> str:
    """获取长文档的树状目录结构（不含正文），用于定位需要阅读的章节。
    doc_id: PageIndex 返回的文档ID（UUID 格式）。
    """
    try:
        result = pageindex_client.get_document_structure(doc_id)
        return result
    except Exception as e:
        return f"获取目录失败: {e}"


@function_tool
def read_long_document_section(doc_id: str, section_id: str) -> str:
    """读取长文档的特定章节内容。
    doc_id: PageIndex 文档ID。
    section_id: 页码/行号范围，如 '5-7' 或 '3,8' 或 '12'。
    """
    try:
        result = pageindex_client.get_page_content(doc_id, section_id)
        return result
    except Exception as e:
        return f"读取章节失败: {e}"


# ── Agent 定义 ──────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """你是一个企业双层知识库领航员。
严禁凭空捏造。每次调用工具前先简述理由。

工作流：
1. 必须优先调用 read_library_directory("SKILL.md") 获取一级目录。
2. 根据用户问题，调用 read_library_directory("{目录名}/INDEX.md") 查看特定分类清单。
3. 发现 [短文档]，调用 get_short_document 获取内容。
4. 发现 [长文档]，严禁猜测内容！必须先调用 view_long_document_toc 获取其内部大纲结构。
5. 结合大纲，调用 read_long_document_section 读取特定小节内容后回答。

回答时引用来源文档。如果信息不在知识库中，明确告知用户。"""

agent = Agent(
    name="CorpusNavigator",
    instructions=AGENT_SYSTEM_PROMPT,
    tools=[read_library_directory, get_short_document, view_long_document_toc, read_long_document_section],
    model=OpenAIChatCompletionsModel(
        model=cfg.AGENT_MODEL,
        openai_client=sf_async_client,
    ),
)


# ── 流式查询 ──────────────────────────────────────────────────────────

async def query_agent(prompt: str) -> str:
    """执行一次 Agent 查询，流式输出到终端。"""
    streamed_run = Runner.run_streamed(agent, prompt)

    async for event in streamed_run.stream_events():
        # 处理原始响应事件（文本流）
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)

        # 处理高级生命周期事件（工具调用等）
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                raw = event.item.raw_item
                print(f"\n[tool call]: {raw.name}({getattr(raw, 'arguments', '{}')})", flush=True)
            elif event.item.type == "tool_call_output_item":
                output = str(event.item.output)
                preview = output[:200] + "..." if len(output) > 200 else output
                print(f"\n[tool call output]: {preview}", flush=True)

    print()  # 末尾换行
    return streamed_run.final_output or ""


def main() -> None:
    """终端交互入口。"""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("  双层知识库导航 Agent（输入 quit/exit 退出）")
    print("=" * 60)

    # 检查编译产物是否存在
    if not cfg.SKILL_MD.exists():
        print("错误：未找到编译产物，请先运行 python -m src.offline_compiler")
        sys.exit(1)

    while True:
        try:
            user_input = input("\n提问> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        try:
            asyncio.run(query_agent(user_input))
        except Exception as e:
            logger.error("查询出错: %s", e, exc_info=True)
            print(f"\n出错了: {e}")


if __name__ == "__main__":
    main()
```

**Step 2: Smoke test**

```bash
cd /Volumes/MyWork/project/agentic_read_agent
echo "如何申请笔记本电脑？" | timeout 30 python -m src.main_agent 2>/dev/null || true
```

Expected: Agent should call `read_library_directory("SKILL.md")` then navigate to find and answer about IT equipment application.

**Step 3: Commit**

```bash
git add src/main_agent.py
git commit -m "feat: add interactive agent with 4-tool two-tier navigation"
```

---

## Task 6: End-to-End Verification

**Step 1: Run full pipeline**

```bash
cd /Volumes/MyWork/project/agentic_read_agent
python -m src.data_generator
python -m src.offline_compiler
```

**Step 2: Verify compiled output structure**

```bash
find compiled_library -type f | sort
```

Expected files:
```
compiled_library/SKILL.md
compiled_library/pageindex_cache/_meta.json
compiled_library/pageindex_cache/<uuid>.json
compiled_library/<cluster1>/INDEX.md
compiled_library/<cluster2>/INDEX.md
compiled_library/short_docs_db.json
```

**Step 3: Interactive test queries**

Run `python -m src.main_agent` and test:

1. "如何申请笔记本电脑？" → 应找到 doc_001 短文档
2. "出差票据丢了怎么办？" → 应导航到长文档的"第五章 票据遗失处理"
3. "VPN 怎么连？" → 应找到 doc_002 短文档
4. "年度体检什么时候？" → 应找到 doc_004 短文档

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete two-tier active navigation RAG system"
```

---

## Summary

| Task | Module | What It Does |
|------|--------|-------------|
| 1 | Environment | .env, pip install, pageindex setup |
| 2 | config.py | Centralized config with fail-fast |
| 3 | data_generator.py | 生成 4 短文档 + 1 长文档 |
| 4 | offline_compiler.py | PageIndex 建树 + Embedding 聚类 + 簇命名 → 物化 |
| 5 | main_agent.py | 4-tool Agent + 流式输出 + 交互循环 |
| 6 | E2E Verify | Full pipeline test with 4 queries |

Total: 6 tasks, ~4 files of implementation code.
