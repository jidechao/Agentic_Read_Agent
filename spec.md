# 项目背景与目标

你好，Claude Code。我需要你帮我从零开始实现一个基于“双层主动导航（Two-Tier Active Navigation）”架构的无向量企业级 RAG 问答系统。
结合《CORPUS2SKILL》论文（离线聚类文件目录）和 PageIndex（长文档树状索引）的思想。

- **Embedding**: 使用 OpenAI 兼容客户端调用 SiliconFlow 的 Qwen3-Embedding-4B (维度 1536)
- **长文档处理**: 调用本地开源 pageindex 库离线建树
- **Agent框架**: 强制使用官方最新的 `openai-agents` SDK (`Agent`, `Runner`, `@function_tool`)

## 🛠️ 第一步：初始化项目与依赖

创建虚拟环境并安装：
`pip install openai openai-agents scikit-learn python-dotenv pageindex numpy requests`

## 📂 第二步：创建基础目录结构

- `data/` 
- `compiled_library/` 
- `src/` 
  - `data_generator.py`
  - `offline_compiler.py` 
  - `main_agent.py` 
- `.env` (含 OPENAI_API_KEY 和 SILICONFLOW_API_KEY)

## 💻 第三步：实现核心代码模块

### 模块 1: `src/data_generator.py`

在 `data/` 下生成：

1. `short_docs.json`：包含 4 篇短文档（包含 `doc_id`, `title`, `content`，涵盖 IT 和 HR 政策）。
2. `travel_policy_2026.md`：生成一篇长文档（涵盖一级、二级标题，含差旅报销和“票据遗失处理”小节）。

### 模块 2: `src/offline_compiler.py` (双层离线编译引擎)

1. **Tier 2 长文档建树:** 使用 `pageindex.PageIndexClient(workspace="compiled_library/pageindex_cache")` 对 `data/travel_policy_2026.md` 执行 `index()`，获取真实的 `doc_id`。提取其前 500 字作为聚类摘要。
2. **提取 Embedding:**
   实例化 SiliconFlow 客户端：`OpenAI(api_key=os.getenv("SILICONFLOW_API_KEY"), base_url="https://api.siliconflow.cn/v1")`。
   调用 `Qwen/Qwen3-Embedding-4B` 对短文 content 和长文摘要进行向量化。
3. **Tier 1 聚类与命名:**
   使用 `KMeans` 对向量聚类 (K=2)。调用 openai (`gpt-4o-mini`) 给每个簇起名（如 `finance-travel`）和写描述。
4. **物化输出:**
   - 写入 `compiled_library/SKILL.md` (总目录)。
   - 写入各子目录的 `INDEX.md`。必须严格按此格式：短文档写为 `- [短文档] {doc_id}: 标题`；长文档写为 `- [长文档] {pageindex的doc_id}: 标题`。
   - 短文档全文存入 `compiled_library/short_docs_db.json`。

### 模块 3: `src/main_agent.py` (在线导航 Agent)

利用 `openai-agents` SDK 编写终端交互程序。

**1. 初始化与工具定义：**
先初始化 `pageindex_client = PageIndexClient(workspace="compiled_library/pageindex_cache")`。
使用 `@function_tool` 定义 4 个工具（提供清晰 docstring）：

- `read_library_directory(path: str)`: 读 `compiled_library` 下的 .md 文件。
- `get_short_document(doc_id: str)`: 读 `short_docs_db.json` 获取全文。
- `view_long_document_toc(doc_id: str)`: 调用 `pageindex_client.get_document_structure(doc_id)`。
- `read_long_document_section(doc_id: str, section_id: str)`: 调 `pageindex_client.get_page_content(doc_id, section_id)`。

**2. Agent 定义：**

```python
AGENT_SYSTEM_PROMPT = """
你是一个企业双层知识库领航员。
严禁凭空捏造。每次调用工具前先简述理由。
工作流：

1. 必须优先调用 read_library_directory("SKILL.md") 获取一级目录。
2. 调用 read_library_directory("{目录名}/INDEX.md") 查看特定分类清单。
3. 发现 [短文档]，调用 get_short_document 获取内容。
4. 发现 [长文档]，严禁猜测内容！必须先调用 view_long_document_toc 获取其内部大纲结构。
5. 结合大纲，调用 read_long_document_section 读取特定小节内容后回答。
   """
   agent = Agent(name="CorpusNavigator", instructions=AGENT_SYSTEM_PROMPT, tools=[...], model="gpt-4o")

**3. 运行与流式输出：**  
实现一个 async def query_agent(prompt: str)，在内部执行 streamed_run = Runner.run_streamed(agent, prompt)。  
请**严格参考** RunItemStreamEvent 和 RawResponsesStreamEvent 的处理逻辑：

- 监听到 tool_call_item 时，高亮打印出形如 [tool call]: 工具名(参数)。

- 监听到 tool_call_output_item 时，截断打印输出结果前 200 个字符：[tool call output]: 结果预览...。

- 监听到 ResponseTextDeltaEvent 时，无缝拼接打印终端回答。  
  最后，在文件底部写一个 while True 获取用户 input()，传入 query_agent 供持续测试。

## 🎯 第四步：执行要求

请保证代码可以直接运行，包含详细中文注释并处理好异常。
