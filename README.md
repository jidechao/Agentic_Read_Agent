# Agentic Read Agent

基于推理的双层导航式 RAG 系统。自动编译企业知识库（PDF/Markdown/DOCX/HTML），通过 LLM Agent 以类人阅读方式检索和回答问题。

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

### 核心组件

| 模块 | 职责 |
|------|------|
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

可选环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CLASSIFIER_TOKEN_THRESHOLD` | 1000 | 长短文档分类 token 阈值 |
| `CLASSIFIER_STRUCTURE_WEIGHT` | 0.4 | 结构评分权重 |
| `CLASSIFIER_LENGTH_WEIGHT` | 0.6 | 长度评分权重 |

### 运行

```bash
# 1. 生成测试数据（可选，也可以放自己的文档到 data/ 目录）
python -m src.data_generator

# 2. 编译知识库
python -m src.triggers.cli compile

# 3. 启动交互式 Agent
python -m src.main_agent
```

Agent 启动后，在终端输入问题即可获得基于知识库的回答。

## 触发模式

```bash
# 手动编译
python -m src.triggers.cli compile

# 查看编译状态
python -m src.triggers.cli status

# 文件监控（data/ 目录变更时自动编译）
python -m src.triggers.cli watch

# 定时调度（每天凌晨 2 点编译）
python -m src.triggers.cli schedule --cron "0 2 * * *"

# HTTP API 服务
python -m src.triggers.cli serve --port 8000
```

API 端点：

- `GET /api/health` — 健康检查
- `POST /api/compile` — 触发编译
- `GET /api/status` — 编译状态
- `GET /api/documents` — 文档列表
- `POST /api/documents/upload` — 上传文档

## 支持的文档格式

| 格式 | 解析方式 | 扩展名 |
|------|----------|--------|
| PDF | PyMuPDF，按字号映射标题层级 | `.pdf` |
| Markdown | 正则提取标题结构 | `.md`, `.markdown` |
| DOCX | python-docx，Heading 样式识别 | `.docx`, `.doc` |
| HTML | BeautifulSoup，h1-h6 标签 | `.html`, `.htm` |

## 技术栈

- **LLM**: SiliconFlow API（Qwen3-30B-A3B）
- **Embedding**: Qwen3-Embedding-4B（1536 维）
- **Agent 框架**: [openai-agents](https://github.com/openai/openai-agents) SDK
- **文档索引**: [PageIndex](https://github.com/jidechao/PageIndex)（基于推理的树结构索引）
- **聚类**: scikit-learn KMeans
- **注册表**: SQLite
- **API 服务**: FastAPI + Uvicorn

## 项目结构

```
├── data/                    # 源文档目录
├── compiled_library/        # 编译产物
│   ├── SKILL.md             # 知识库总目录
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
│       ├── cli.py
│       ├── watcher.py
│       ├── scheduler.py
│       └── api_server.py
├── tests/                   # 测试（53 tests）
├── PageIndex/               # PageIndex 子模块
└── knowledge.db             # 文档注册表数据库
```

## License

MIT
