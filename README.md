# Agentic Read Agent

基于推理的双层导航式 RAG 系统。自动编译企业知识库（PDF/Markdown/DOCX/HTML），通过 LLM Agent 以类人阅读方式检索和回答问题。

## 目录

- [架构](#架构)
- [快速开始](#快速开始)
- [配置详解](#配置详解)
- [编译系统](#编译系统)
- [Agent 使用](#agent-使用)
- [触发模式](#触发模式)
- [支持的文档格式](#支持的文档格式)
- [技术栈](#技术栈)
- [项目结构](#项目结构)

## 架构

```
┌─────────────────────────────────────────────────────┐
│                  用户提问                            │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│            LLM Agent (openai-agents)                │
│  1. 读取 SKILL.md 获取知识库目录                     │
│  2. 扫描所有分类的 INDEX.md                          │
│  3. 按需获取文档内容（短文档全文 / 长文档逐节导航）    │
│  4. 基于实际内容回答问题                              │
└──────────┬───────────────────────────┬──────────────┘
           ▼                           ▼
┌────────────────────┐    ┌────────────────────────────┐
│    短文档层         │    │       长文档层              │
│  JSON 键值存储      │    │  PageIndex 树索引           │
│  直接全文读取       │    │  Agent 先读大纲             │
│                    │    │  再按 line_num 读具体章节    │
└────────────────────┘    └────────────────────────────┘
```

### 编译流程

```
源文档目录 (data/)                    编译产物 (compiled_library/)
     │                                       │
     ▼                                       ▼
  扫描文件 ──SHA256──▶ 变更检测        SKILL.md (总目录)
     │                 │                     │
     ▼                 ▼                     ▼
  新增/变更文件     跳过未变文件       <cluster>/INDEX.md
     │                                       │
     ▼                                       ▼
  解析 ──▶ 分类 ──▶ 嵌入 ──▶ 聚类     pageindex_cache/
                 │        │              short_docs_db.json
                 │        │
                 │        └─ 标题+章节+摘要（非全文，避免语义稀释）
                 │
                 └─ 自动选择 K（轮廓系数 + 粒度保护）
                               ▼
                         原子物化 (tmp→bak→rename)
```

### 文档生命周期

每个文档在 SQLite 注册表中经历以下状态：

```
ingested → classified → embedded → compiled
                                  ↘ error (可重试)
```

### 核心组件

| 模块 | 职责 |
|------|------|
| `src/config.py` | 统一配置管理，环境变量 + 默认值 |
| `src/ingester.py` | 多格式文档解析（PDF/MD/DOCX/HTML） |
| `src/classifier.py` | 自动分类：结构评分 + 长度评分 → 短文档/长文档 |
| `src/compiler.py` | 增量编译引擎：扫描 → 解析 → 分类 → 嵌入 → 聚类 |
| `src/materializer.py` | 原子物化：tmp→bak→rename 写入 compiled_library/ |
| `src/registry.py` | SQLite 文档注册表：状态机 + 嵌入缓存 + 编译追踪 |
| `src/main_agent.py` | 交互式 Agent：openai-agents SDK + 流式输出 |
| `src/triggers/` | 4 种触发模式：CLI / 文件监控 / 定时调度 / HTTP API |

## 快速开始

### 前置条件

- Python 3.11+
- [SiliconFlow](https://siliconflow.cn) API Key（用于 LLM 和 Embedding 调用）

### 安装

```bash
git clone --recurse-submodules https://github.com/jidechao/Agentic_Read_Agent.git
cd Agentic_Read_Agent

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r PageIndex/requirements.txt
```

### 配置

创建 `.env` 文件：

```env
SILICONFLOW_API_KEY=sk-your-api-key-here
```

### 运行

```bash
# 1. 生成测试数据（可选，也可以放自己的文档到 data/ 目录）
python -m src.data_generator

# 2. 编译知识库
python -m src.triggers.cli compile

# 3. 启动交互式 Agent
python -m src.main_agent
```

Agent 启动后，在终端输入问题即可获得基于知识库的回答。输入 `quit` 或 `exit` 退出。

## 配置详解

所有配置集中在 `src/config.py`，通过环境变量覆盖默认值。

### API 配置

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| `SILICONFLOW_API_KEY` | `SILICONFLOW_API_KEY` | （必填） | SiliconFlow API 密钥，用于 LLM 推理和 Embedding 生成 |
| `SILICONFLOW_BASE_URL` | — | `https://api.siliconflow.cn/v1` | SiliconFlow API 基础 URL |

> **注意**：`SILICONFLOW_API_KEY` 为必填项，缺失时程序启动即报错退出。

### 模型配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-4B` | 文档嵌入模型，用于聚类和相似度计算 |
| `EMBEDDING_DIMENSION` | `1536` | 嵌入向量维度，需与模型匹配 |
| `AGENT_MODEL` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 在线 Agent 使用的对话模型 |
| `CLUSTER_NAMING_MODEL` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 聚类命名模型，为自动分类生成目录名和描述 |
| `PAGEINDEX_MODEL` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | PageIndex 长文档索引模型，用于生成章节摘要 |

> **兼容性**：系统使用 SiliconFlow API，不支持 OpenAI Responses API。Agent 通过 `OpenAIChatCompletionsModel` 适配。PageIndex 内部使用 LiteLLM，模型名自动添加 `openai/` 前缀路由。

### 路径配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DATA_DIR` | `项目根目录/data/` | 源文档目录，编译时扫描此目录 |
| `COMPILED_DIR` | `项目根目录/compiled_library/` | 编译产物输出目录 |
| `PAGEINDEX_WORKSPACE` | `compiled_library/pageindex_cache/` | PageIndex 索引缓存目录 |
| `SHORT_DOCS_DB` | `compiled_library/short_docs_db.json` | 短文档内容数据库 |
| `SKILL_MD` | `compiled_library/SKILL.md` | 知识库总目录文件 |
| `KNOWLEDGE_DB` | `项目根目录/knowledge.db` | SQLite 文档注册表数据库 |

### 自动分类器配置

分类器通过 **结构评分 × 权重 + 长度评分 × 权重** 的加权公式决定文档归类：

- `final_score = structure_score × STRUCTURE_WEIGHT + length_score × LENGTH_WEIGHT`
- `> 0.5` → 长文档（PageIndex 树索引）
- `≤ 0.5` → 短文档（JSON 键值存储）

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| `CLASSIFIER_TOKEN_THRESHOLD` | `CLASSIFIER_TOKEN_THRESHOLD` | `1000` | 长度评分的基准阈值（token 数）。文档 token 数 ≥ 此值时 length_score = 1.0 |
| `CLASSIFIER_STRUCTURE_WEIGHT` | `CLASSIFIER_STRUCTURE_WEIGHT` | `0.4` | 结构评分权重。结构评分基于标题层级深度和数量 |
| `CLASSIFIER_LENGTH_WEIGHT` | `CLASSIFIER_LENGTH_WEIGHT` | `0.6` | 长度评分权重。长度评分 = token 数 / THRESHOLD |
| `CLASSIFIER_USE_EXACT_TOKENS` | `CLASSIFIER_USE_EXACT_TOKENS` | `false` | 是否使用精确 token 计数（需 tiktoken），默认用估算 |

> **调参示例**：如果你的文档普遍较长但仍想作为短文档处理，可增大 `CLASSIFIER_TOKEN_THRESHOLD` 到 2000；如果更倾向使用 PageIndex 索引，可降低到 500。

### 聚类配置

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| `CLUSTER_K` | `CLUSTER_K` | `0`（自动） | 聚类数。设为 0 时通过轮廓系数自动选择最优 K；低分时偏向更细粒度 |
| `LONG_DOC_SUMMARY_LENGTH` | — | `500` | 长文档用于聚类时的摘要截取字数 |

> **自动 K 选择**：系统尝试 K=2 到 K=min(n-1, √n×2) 的所有值，计算轮廓系数（silhouette score），选最优 K。当所有分数较低（<0.15）时，倾向更多聚类以确保每个分类更聚焦。

## 编译系统

### 编译命令

```bash
# 增量编译（默认）— 只处理新增或变更的文件
python -m src.triggers.cli compile

# 指定源文档目录
python -m src.triggers.cli compile --data-dir /path/to/docs

# 强制全量重编译
python -m src.triggers.cli compile --force

# 查看当前编译状态
python -m src.triggers.cli status
```

### 增量编译机制

编译引擎通过 SHA256 哈希值实现增量编译，避免重复处理：

| 场景 | 处理方式 |
|------|----------|
| 新文件（hash 不在 registry 中） | 完整处理：解析 → 分类 → 嵌入 → 索引 |
| 未变更文件（hash 匹配且 status=compiled） | 跳过 |
| 变更文件（hash 已存在但内容不同） | 重新处理 |
| 已删除文件（registry 中有但目录中找不到） | 从 registry 中删除 |
| 错误文件（status=error） | 自动重试 |

### 编译产物

编译完成后，`compiled_library/` 目录结构如下：

```
compiled_library/
├── SKILL.md                  # 知识库总目录（Agent 第一步读取）
├── short_docs_db.json        # 短文档全文数据
├── pageindex_cache/          # PageIndex 长文档索引缓存
│   ├── _meta.json
│   └── <doc-uuid>.json
└── <cluster-name>/           # 自动生成的分类目录
    └── INDEX.md              # 分类内文档列表
```

### 原子物化

编译产物通过 tmp→bak→rename 三步原子写入，确保任何时候 `compiled_library/` 目录都处于完整可用状态。即使编译中途失败，Agent 仍可使用上一次的成功产物。

### 文档自动分类

编译时自动判断每个文档的存储策略：

- **短文档**：内容较短（token < 阈值）且结构简单的文档，全文存入 JSON，Agent 直接读取
- **长文档**：内容较长或结构复杂的文档，通过 PageIndex 构建树索引，Agent 逐节导航读取

## Agent 使用

### 交互模式

```bash
python -m src.main_agent
```

启动后进入交互式问答界面：

```
============================================================
  双层知识库导航 Agent（输入 quit/exit 退出）
============================================================

提问> 介绍一下pageindex
```

### Agent 工作流程

1. 读取 `SKILL.md` 获取知识库所有分类目录
2. 扫描每个分类的 `INDEX.md`，通过文档标题判断相关性
3. 获取相关文档内容：
   - **短文档**：直接读取全文
   - **长文档**：先读取树状大纲，再按 `line_num` 定位并读取具体章节
4. 基于工具返回的实际内容回答问题，引用来源

### Agent 工具

| 工具 | 功能 |
|------|------|
| `read_library_directory` | 读取 compiled_library 下的文件 |
| `get_short_document` | 获取短文档全文 |
| `view_long_document_toc` | 获取长文档树状目录（含 line_num） |
| `read_long_document_section` | 按 line_num 读取长文档章节 |
| `get_document_source` | 查询文档原始来源信息 |

## 触发模式

### CLI 手动触发

```bash
python -m src.triggers.cli compile [--force] [--data-dir PATH] [-v]
python -m src.triggers.cli status
```

### 文件监控

监控 `data/` 目录变化，文件新增/修改/删除时自动触发编译（0.5 秒防抖）：

```bash
python -m src.triggers.cli watch [--dir /path/to/watch]
```

### 定时调度

基于 APScheduler 的 cron 定时编译：

```bash
# 每天凌晨 2 点
python -m src.triggers.cli schedule --cron "0 2 * * *"

# 每小时
python -m src.triggers.cli schedule --cron "0 * * * *"
```

### HTTP API

```bash
python -m src.triggers.cli serve [--host 0.0.0.0] [--port 8000]
```

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/compile` | POST | 触发编译（支持 `force=true`） |
| `/api/status` | GET | 最近编译状态 |
| `/api/documents` | GET | 文档列表 |
| `/api/documents/upload` | POST | 上传文档（multipart/form-data） |

## 支持的文档格式

| 格式 | 解析方式 | 扩展名 | 标题层级识别 |
|------|----------|--------|-------------|
| PDF | PyMuPDF span 级提取 | `.pdf` | 按字号映射：≥22→L1, ≥14.5→L2, ≥12.5→L3 |
| Markdown | 正则 `#{1,6}` 提取 | `.md`, `.markdown` | 保留原始 `#` 层级 |
| DOCX | python-docx Heading 样式 | `.docx`, `.doc` | Heading 1-6 映射 |
| HTML | BeautifulSoup h1-h6 标签 | `.html`, `.htm` | 保留原始标签层级 |

> **PDF 解析优化**：自动过滤 emoji-only 和纯数字标题，避免导航干扰。

## 技术栈

| 层面 | 技术 | 用途 |
|------|------|------|
| LLM | SiliconFlow API (Qwen3-30B-A3B) | Agent 对话、聚类命名、PageIndex 摘要 |
| Embedding | Qwen3-Embedding-4B (1536 维) | 文档向量化、KMeans 聚类 |
| Agent 框架 | [openai-agents](https://github.com/openai/openai-agents) SDK | 工具调用、流式输出 |
| 文档索引 | [PageIndex](https://github.com/jidechao/PageIndex) | 长文档树结构索引、逐节检索 |
| 文档解析 | PyMuPDF / python-docx / BeautifulSoup | PDF / DOCX / HTML 解析 |
| 聚类 | scikit-learn KMeans + silhouette score | 文档自动分类（动态 K） |
| 注册表 | SQLite | 文档状态管理、嵌入缓存 |
| API 服务 | FastAPI + Uvicorn | HTTP 触发接口 |
| 文件监控 | watchdog | 目录变更检测 |

## 项目结构

```
├── data/                    # 源文档目录
├── compiled_library/        # 编译产物
│   ├── SKILL.md             # 知识库总目录
│   ├── short_docs_db.json   # 短文档全文
│   ├── pageindex_cache/     # PageIndex 索引缓存
│   └── <cluster>/           # 分类目录
│       └── INDEX.md         # 分类文档索引
├── src/
│   ├── config.py            # 统一配置
│   ├── ingester.py          # 多格式文档解析
│   ├── classifier.py        # 自动分类器
│   ├── compiler.py          # 增量编译引擎
│   ├── materializer.py      # 原子物化
│   ├── registry.py          # SQLite 注册表
│   ├── main_agent.py        # 交互式 Agent
│   ├── data_generator.py    # 测试数据生成
│   └── triggers/            # 触发模式
│       ├── cli.py           # 命令行接口
│       ├── watcher.py       # 文件监控
│       ├── scheduler.py     # 定时调度
│       └── api_server.py    # HTTP API
├── tests/                   # 测试（53 tests）
├── PageIndex/               # PageIndex 子模块
├── knowledge.db             # 文档注册表数据库
├── requirements.txt         # Python 依赖
├── CLAUDE.md                # Claude Code 开发指引
└── .env                     # 环境变量（不入库）
```

## License

MIT
