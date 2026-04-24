# 生产级知识库离线编译系统设计

> 日期: 2026-04-24
> 状态: 已批准

## 1. 背景与目标

当前离线编译系统（`src/offline_compiler.py`）不具备生产特性：
- 数据硬编码在 `data_generator.py` 中，不支持真实文档接入
- 仅支持 Markdown 格式，无法处理 PDF/DOCX/HTML
- 无增量编译，每次全量重跑 embedding + 聚类
- 无原子性输出，编译中途崩溃留下不完整产物
- 无错误恢复，API 调用失败直接终止
- 无回查机制，Agent 无法追溯文档原始来源

**目标**：设计一个生产级知识库编译系统，支持多格式文档接入、自动分类、增量编译、原子输出、四种触发模式，并与现有在线搜索 Agent 完全兼容。

## 2. 方案选型

在三个方案中选择了 **方案 B：SQLite 注册表管线**：
- JSON manifest（方案 A）轻量但并发和查询能力有限
- 插件化架构（方案 C）过度设计
- SQLite 提供 ACID 保证、零配置、Python 内置支持，适合单机优先可演进的策略

## 3. 整体架构

```
src/
  config.py              ← 扩展配置项（分类阈值、触发模式参数）
  registry.py            ← SQLite 文档注册表
  ingester.py            ← 文档接入层（格式解析）
  classifier.py          ← 自动长短分类器
  compiler.py            ← 增量编译引擎（替代原 offline_compiler.py）
  materializer.py        ← 原子物化输出
  triggers/              ← 触发模式
    __init__.py
    cli.py               ← 手动 CLI
    watcher.py           ← 文件监控（watchdog）
    scheduler.py         ← 定时任务（APScheduler）
    api_server.py        ← FastAPI 端点
  main_agent.py          ← 现有 Agent，增量改动

knowledge.db             ← SQLite 注册表（新增）
data/                    ← 原始文档目录（支持多格式文件）
compiled_library/        ← 编译产物（结构不变）
```

### 管线流程

```
文档文件 → [Ingest] → 纯文本+结构元数据
         → [Classify] → tier=short | tier=long
         → [Embed & Cache] → 向量存入 SQLite
         → [Compile] → 增量聚类 + PageIndex 建树
         → [Materialize] → 原子写入 compiled_library/
```

每步通过 SQLite 记录状态，支持断点续跑。

## 4. SQLite 注册表 Schema

### 4.1 文档表

```sql
CREATE TABLE documents (
    id            TEXT PRIMARY KEY,          -- UUID
    source_path   TEXT NOT NULL,             -- 原始文件路径（相对路径）
    format        TEXT NOT NULL,             -- pdf / md / docx / html
    title         TEXT,                      -- 文档标题
    content_hash  TEXT NOT NULL,             -- SHA256，变更检测
    tier          TEXT CHECK(tier IN ('short', 'long', 'unclassified')),
    status        TEXT CHECK(status IN ('ingested', 'classified', 'embedded', 'compiled', 'error')),
    token_count   INTEGER,                  -- 文档 token 数
    has_structure BOOLEAN DEFAULT FALSE,     -- 是否有章节结构
    embedding     BLOB,                      -- 向量缓存（numpy → bytes）
    pageindex_id  TEXT,                      -- PageIndex doc_id（仅长文档）
    source_page_hint TEXT,                   -- 原始页码提示
    error_message TEXT,                      -- 最近错误信息
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### 4.2 编译记录表

```sql
CREATE TABLE compile_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_type    TEXT,                     -- manual / watcher / scheduler / api
    started_at      DATETIME NOT NULL,
    completed_at    DATETIME,
    status          TEXT CHECK(status IN ('running', 'completed', 'failed')),
    docs_processed  INTEGER DEFAULT 0,
    docs_skipped    INTEGER DEFAULT 0,
    error_message   TEXT
);
```

### 4.3 聚类结果表

```sql
CREATE TABLE clusters (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    description   TEXT,
    compile_run_id INTEGER REFERENCES compile_runs(id)
);
```

### 4.4 文档-聚类关联表

```sql
CREATE TABLE document_clusters (
    doc_id      TEXT REFERENCES documents(id),
    cluster_id  INTEGER REFERENCES clusters(id),
    PRIMARY KEY (doc_id, cluster_id)
);
```

### 4.5 文档状态机

```
ingested → classified → embedded → compiled
                         ↑            ↑
                      断点续跑      断点续跑
任意阶段可进入 error 状态（记录 error_message）
```

## 5. 文档接入层（Ingester）

### 5.1 统一接口

```python
class DocumentIngester:
    def ingest(self, file_path: Path) -> IngestResult: ...

@dataclass
class IngestResult:
    text: str                    # 纯文本内容
    title: str                   # 文档标题
    headings: list[Heading]      # 标题层级
    has_tables: bool             # 包含表格
    has_lists: bool              # 包含列表
    metadata: dict               # 格式特有元数据
```

### 5.2 格式解析器选型

| 格式 | 库 | 提取内容 |
|------|-----|---------|
| PDF | PyMuPDF (`fitz`) | 文本 + 标题（字号推断）+ 表格 |
| Markdown | `mistune` 或内置 | AST 解析标题层级 |
| DOCX | `python-docx` | 段落 + 标题样式 + 表格 |
| HTML | `BeautifulSoup4` | h1-h6 + p + table + list |

所有格式统一输出 `IngestResult`，下游不感知原始格式。

## 6. 自动分类器

### 6.1 分类策略

结构 + 长度组合判断，权重可配：

```python
# config.py 新增
CLASSIFIER_TOKEN_THRESHOLD: int = 1000     # token 阈值
CLASSIFIER_STRUCTURE_WEIGHT: float = 0.4   # 结构权重
CLASSIFIER_LENGTH_WEIGHT: float = 0.6      # 长度权重
CLASSIFIER_USE_EXACT_TOKENS: bool = False   # True=精确计数, False=字符估算
```

### 6.2 Token 计数

- 默认：字符估算（中文 ≈ 1.5 token/字，英文按空格分词），零额外依赖
- 精确模式：调用 embedding API 的 tokenizer

### 6.3 分类规则

- 有明确章节结构（h1/h2/h3 层级 ≥ 2 层）→ 长文档候选
- token 数超过阈值 → 长文档候选
- 加权评分：`structure_score * 0.4 + length_score * 0.6 > 0.5` → 长文档
- 其余 → 短文档

## 7. 增量编译引擎

### 7.1 变更检测

- SHA256 hash 对比：hash 不变且 status=compiled → 跳过
- hash 变化 → 重置 status 为 ingested，重新处理
- 新文件 → 新增处理
- 文件已删除但 SQLite 有记录 → 标记删除，触发重编译

### 7.2 Embedding 缓存

向量存为 BLOB in SQLite，只对新增/变更文档调 API。

### 7.3 聚类重算

- 仅文档集合变化时重算 KMeans
- 聚类数 K 自动推断：`K = max(1, round(sqrt(N / 2)))`
- 用户可通过 `CLUSTER_K` 配置覆盖

### 7.4 PageIndex 增量

- 长文档 hash 不变则跳过建树（复用 PageIndex 自身缓存）
- hash 变化则重建

## 8. 原子物化

```
compiled_library.tmp/   ← 编译期间写入
compiled_library.bak/   ← 上一版备份
compiled_library/       ← 当前生效产物

流程: 写 .tmp → 成功后 rename 当前为 .bak → rename .tmp 为 compiled_library/
```

## 9. 触发模式

四种模式共享编译核心 `compiler.compile(trigger_type)`：

### 9.1 手动 CLI

```bash
python -m src.compiler compile              # 增量（默认）
python -m src.compiler compile --force       # 全量
python -m src.compiler status                # 查看状态
```

### 9.2 文件监控（watchdog）

```bash
python -m src.compiler watch --dir data/
```

500ms 防抖，连续事件合并为一次编译。

### 9.3 定时调度（APScheduler）

```bash
python -m src.compiler schedule --cron "0 2 * * *"
```

支持标准 cron 表达式。

### 9.4 API 端点（FastAPI）

```bash
python -m src.compiler serve --port 8000
```

端点：
- `POST /api/compile` → 触发编译
- `GET /api/status` → 注册表状态
- `GET /api/documents` → 文档列表
- `POST /api/documents/upload` → 上传并编译
- `GET /api/health` → 健康检查

## 10. 回查机制

### 10.1 编译产物中嵌入来源

INDEX.md 扩展格式：
```markdown
- [短文档] doc_001: IT设备申请流程 (来源: data/policies/it_equipment.pdf)
- [长文档] uuid-xxx: 差旅报销政策 (来源: data/policies/travel_2026.pdf, PageIndex索引)
```

### 10.2 Agent 新增工具

```python
@function_tool
def get_document_source(doc_id: str) -> str:
    """查询文档原始来源信息（路径、格式、页码提示）。"""
```

### 10.3 路径可移植性

`source_path` 存相对路径，Agent 运行时拼接绝对路径。

## 11. 在线搜索兼容性

### 11.1 不变部分

| 组件 | 说明 |
|------|------|
| `read_library_directory` | 读 compiled_library/*.md，格式不变 |
| `view_long_document_toc` | PageIndex client，同一 workspace |
| `read_long_document_section` | PageIndex client，同一 workspace |
| Agent routing prompt | SKILL.md 结构不变，关键词映射不变 |
| PageIndex workspace | `compiled_library/pageindex_cache/` 不变 |

### 11.2 增量改动

| 组件 | 变化 |
|------|------|
| `get_short_document` | 从 JSON 文件改为 SQLite 查询（大文档量友好） |
| 新增 `get_document_source` | 查询文档原始来源 |
| Agent 初始化 | 新增 SQLite 连接 |

### 11.3 长文档搜索链路

编译阶段：PDF/DOCX/HTML → Ingester 转纯文本 → 写入临时 MD → PageIndex 建树
在线阶段：Agent → PageIndex TOC → PageIndex section（完全不变）

## 12. 错误处理

- API 调用失败：指数退避重试（1s → 2s → 4s，最多 3 次）
- 最终失败：文档 status 设为 error，记录 error_message，不阻塞其他文档
- 结构化日志：每模块独立 logger，编译摘要自动输出

## 13. 新增依赖

| 包 | 用途 | 必选/可选 |
|----|------|----------|
| PyMuPDF | PDF 解析 | 必选 |
| python-docx | DOCX 解析 | 必选 |
| beautifulsoup4 + lxml | HTML 解析 | 必选 |
| mistune | Markdown AST 解析 | 必选 |
| watchdog | 文件监控触发 | 可选 |
| apscheduler | 定时调度触发 | 可选 |
| fastapi + uvicorn | API 触发 | 可选 |
