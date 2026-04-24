# Two-Tier Active Navigation RAG - Design Document

**Date**: 2026-04-23
**Status**: Approved
**Scope**: 生产级 MVP

## 1. Overview

基于"双层主动导航（Two-Tier Active Navigation）"架构的无向量企业级 RAG 问答系统。结合 CORPUS2SKILL（离线聚类文件目录）和 PageIndex（长文档树状索引）思想，所有 LLM 调用统一走 SiliconFlow。

## 2. Architecture

### Data Flow

```
data_generator.py → data/ → offline_compiler.py → compiled_library/ → main_agent.py
```

1. **data_generator.py**: 生成测试数据（4 短文档 + 1 长文档）
2. **offline_compiler.py**: 双层编译 — PageIndex 建树 + Embedding 聚类 + 簇命名
3. **main_agent.py**: 在线 Agent，4 工具导航双层知识库

### Directory Structure

```
src/
  config.py              # 统一配置
  data_generator.py      # 生成测试数据
  offline_compiler.py    # 双层离线编译引擎
  main_agent.py          # 在线导航 Agent
data/                    # 原始数据（由 data_generator 生成）
compiled_library/        # 编译产物（由 offline_compiler 生成）
.env                     # SILICONFLOW_API_KEY
```

### Short vs Long Document Classification

| 维度 | 短文档 | 长文档 |
|------|--------|--------|
| 数据源 | `data/short_docs.json`（结构化 JSON） | `data/travel_policy_2026.md`（Markdown） |
| 判断 | 从 short_docs.json 读入 = 短 | 独立 .md 文件 = 长，走 PageIndex 建树 |
| 存储 | `compiled_library/short_docs_db.json` | `compiled_library/pageindex_cache/` |
| Agent 标识 | `[短文档] doc_id: 标题` | `[长文档] pageindex_doc_id: 标题` |

后续扩展可加长度阈值自动判断，当前 MVP 数据量固定，数据源驱动分流。

## 3. SiliconFlow Integration

### Configuration (src/config.py)

```python
SILICONFLOW_BASE_URL     = "https://api.siliconflow.cn/v1"
EMBEDDING_MODEL          = "Qwen/Qwen3-Embedding-4B"
EMBEDDING_DIMENSION      = 1536
AGENT_MODEL              = "Qwen/Qwen3-32B"    # 可配置
CLUSTER_NAMING_MODEL     = "Qwen/Qwen3-8B"     # 可配置
PAGEINDEX_WORKSPACE      = "compiled_library/pageindex_cache"
DATA_DIR                 = "data"
COMPILED_DIR             = "compiled_library"
CLUSTER_K                = 2
```

### Three LLM Integration Points

| 场景 | 客户端 | 方式 |
|------|--------|------|
| Embedding 向量化 | `OpenAI(base_url=SF_URL, api_key=SF_KEY)` | `embeddings.create()` |
| 聚类命名 | 同上 | `chat.completions.create()` |
| Agent 推理 | `openai-agents` SDK | `OpenAIChatCompletionsModel` + `set_default_openai_client` |

### Agent Initialization Pattern

```python
from openai import AsyncOpenAI
from agents import set_default_openai_client, Agent
from agents.models import OpenAIChatCompletionsModel

sf_client = AsyncOpenAI(
    base_url=config.SILICONFLOW_BASE_URL,
    api_key=os.getenv("SILICONFLOW_API_KEY")
)
set_default_openai_client(sf_client)

agent = Agent(
    name="CorpusNavigator",
    instructions=AGENT_SYSTEM_PROMPT,
    tools=[...],
    model=OpenAIChatCompletionsModel(
        model=config.AGENT_MODEL,
        openai_client=sf_client
    )
)
```

使用 `OpenAIChatCompletionsModel` 而非默认 Responses API，因 SiliconFlow 仅兼容 `/v1/chat/completions`。

### PageIndex SiliconFlow Configuration

通过环境变量传导：

```python
os.environ["OPENAI_API_KEY"] = os.getenv("SILICONFLOW_API_KEY")
os.environ["OPENAI_BASE_URL"] = config.SILICONFLOW_BASE_URL

pageindex_client = PageIndexClient(
    workspace=config.PAGEINDEX_WORKSPACE,
    model="Qwen/Qwen3-8B"
)
```

PageIndex 相关代码（初始化、索引、检索）参考 `agentic_vectorless_rag_demo.py` 已验证模式。

## 4. Offline Compiler (offline_compiler.py)

### Execution Flow

1. 读取 `data/short_docs.json` → 短文档列表
2. 读取 `data/travel_policy_2026.md` → 长文档
3. **长文档建树**: `pageindex_client.index("data/travel_policy_2026.md")` → doc_id，提取前 500 字为聚类摘要
4. **向量化**: SiliconFlow Qwen3-Embedding-4B 对短文档 content + 长文档摘要生成向量
5. **聚类**: `KMeans(n_clusters=2).fit_predict(vectors)` → labels
6. **簇命名**: 对每个簇的文档标题调用 LLM 生成 name + description
7. **物化输出**: 写入 SKILL.md、各 INDEX.md、short_docs_db.json

### SKILL.md Format

```markdown
# 知识库总目录

## [finance-travel](finance-travel/INDEX.md)
差旅报销与财务政策相关文档

## [it-hr](it-hr/INDEX.md)
IT设备与人力资源政策文档
```

### INDEX.md Format

```markdown
# finance-travel

- [短文档] doc_003: 出差住宿标准
- [长文档] a1b2c3d4-e5f6-...: 2026年差旅报销政策
```

## 5. Online Agent (main_agent.py)

### 4 Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `read_library_directory` | `path: str` | 读 compiled_library 下的 .md 文件 |
| `get_short_document` | `doc_id: str` | 从 short_docs_db.json 获取全文 |
| `view_long_document_toc` | `doc_id: str` | 获取长文档树状目录 |
| `read_long_document_section` | `doc_id: str, section_id: str` | 读取特定小节内容 |

### Agent System Prompt

```
你是企业双层知识库领航员。
严禁凭空捏造。每次调用工具前先简述理由。
工作流：
1. 必须优先调用 read_library_directory("SKILL.md") 获取一级目录。
2. 调用 read_library_directory("{目录名}/INDEX.md") 查看特定分类清单。
3. 发现 [短文档]，调用 get_short_document 获取内容。
4. 发现 [长文档]，严禁猜测内容！必须先调用 view_long_document_toc 获取其内部大纲结构。
5. 结合大纲，调用 read_long_document_section 读取特定小节内容后回答。
```

### Streaming Output

沿用 demo 已验证的事件处理模式：

| 事件 | 处理 | 输出格式 |
|------|------|---------|
| `tool_call_item` | 打印工具调用 | `[tool call]: read_library_directory(SKILL.md)` |
| `tool_call_output_item` | 截断前200字 | `[tool call output]: 结果预览...` |
| `ResponseTextDeltaEvent` | 流式拼接 | 直接输出终端文本 |

### Interaction Loop

```python
if __name__ == "__main__":
    while True:
        user_input = input("提问> ")
        if user_input.lower() in ("quit", "exit"): break
        asyncio.run(query_agent(user_input))
```

## 6. Dependencies

```
openai
openai-agents
scikit-learn
python-dotenv
pageindex  (本地: PageIndex/)
numpy
```

## 7. Error Handling Strategy

- 所有 SiliconFlow API 调用：try/except + 重试（网络超时）
- PageIndex 索引：检查文件存在性，处理空结构
- Agent 工具：返回明确错误信息而非抛异常
- 配置缺失：启动时 fail-fast，明确提示缺少哪个环境变量
