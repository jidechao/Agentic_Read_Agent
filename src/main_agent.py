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
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

import src.config as cfg
from src.registry import KnowledgeRegistry

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

# 3) 初始化知识库注册表（如果存在）
_knowledge_registry: KnowledgeRegistry | None = None
if cfg.KNOWLEDGE_DB.exists():
    _knowledge_registry = KnowledgeRegistry(cfg.KNOWLEDGE_DB)

# 4) 加载短文档库
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
    # 优先从 SQLite 查询
    if _knowledge_registry:
        doc = _knowledge_registry.get_document(doc_id)
        if doc and doc.get("tier") == "short":
            # 尝试从 compiled_library/short_docs_db.json 读取完整内容
            db_path = cfg.COMPILED_DIR / "short_docs_db.json"
            if db_path.exists():
                db = json.loads(db_path.read_text(encoding="utf-8"))
                entry = db.get(doc_id)
                if entry:
                    return json.dumps(entry, ensure_ascii=False)
            # 降级：返回注册表中的基本信息
            return json.dumps({
                "doc_id": doc["id"],
                "title": doc["title"],
                "source_path": doc["source_path"],
                "status": doc["status"],
            }, ensure_ascii=False)
    # 降级到 JSON 文件（向后兼容）
    doc = _short_docs_db.get(doc_id)
    if not doc:
        available = ", ".join(_short_docs_db.keys()) if _short_docs_db else "无"
        return f"错误：未找到文档 {doc_id}。可用文档: {available}"
    return json.dumps(doc, ensure_ascii=False)


@function_tool
def view_long_document_toc(doc_id: str) -> str:
    """获取长文档的树状目录结构（含每个节点的 line_num），用于定位需要阅读的章节。
    doc_id: PageIndex 返回的文档ID（UUID 格式）。
    返回结果中每个节点包含 line_num 字段，读取章节时必须使用该值作为 section_id。
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
    section_id: 必须使用 view_long_document_toc 返回的 line_num 值。
      支持格式：单个行号 '50'、行号范围 '30-67'、多个行号 '30,45,67'。
      注意：不是 node_id！必须从目录结构中找到目标章节的 line_num 字段值。
    """
    try:
        result = pageindex_client.get_page_content(doc_id, section_id)
        return result
    except Exception as e:
        return f"读取章节失败: {e}"


@function_tool
def get_document_source(doc_id: str) -> str:
    """查询文档的原始来源信息（文件路径、格式、页码提示）。
    doc_id: 文档ID。
    """
    if not _knowledge_registry:
        return "知识库注册表未初始化，无法查询来源信息。"
    doc = _knowledge_registry.get_document(doc_id)
    if not doc:
        return f"错误：未找到文档 {doc_id}"
    info = {
        "doc_id": doc["id"],
        "source_path": doc["source_path"],
        "format": doc["format"],
        "title": doc["title"],
        "tier": doc["tier"],
    }
    if doc.get("source_page_hint"):
        info["page_hint"] = doc["source_page_hint"]
    if doc.get("pageindex_id"):
        info["pageindex_id"] = doc["pageindex_id"]
    return json.dumps(info, ensure_ascii=False)


# ── Agent 定义 ──────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """你是一个企业双层知识库领航员。严禁凭空捏造。

严格按以下工作流执行，每一步都必须调用工具，不要跳步：

第一步：调用 read_library_directory("SKILL.md") 获取所有分类目录。

第二步：**必须查看所有分类目录的 INDEX.md**，逐一调用 read_library_directory 查看每个分类的文档标题列表。
不要根据目录名猜测！必须读取每个 INDEX.md 的实际内容，通过文档标题判断哪些文档与用户问题相关。

第三步：根据 INDEX.md 中找到的相关文档获取内容：
- 条目标记为 [短文档]：调用 get_short_document(doc_id) 获取全文
- 条目标记为 [长文档]：严禁猜测内容！必须按以下两步操作：
  a) 先调用 view_long_document_toc(doc_id) 获取大纲结构（doc_id 是 INDEX.md 中长文档后面那串UUID）
  b) 根据大纲找到相关章节，读取该节点的 line_num 值，调用 read_long_document_section(doc_id, section_id)
     section_id 必须使用目录中节点的 line_num 值（不是 node_id！）
     例如：节点 line_num=30，则传 section_id="30"；跨越多个节点则用范围 "30-45"

第四步：基于工具返回的实际内容回答用户问题，引用来源文档。

重要提醒：
- 第二步必须查看所有分类，不要跳过任何目录
- 长文档的 doc_id 是一串 UUID，必须从 INDEX.md 中复制，严禁编造
- 一次回答可能需要调用多个工具"""

agent = Agent(
    name="CorpusNavigator",
    instructions=AGENT_SYSTEM_PROMPT,
    tools=[read_library_directory, get_short_document, view_long_document_toc, read_long_document_section, get_document_source],
    model=OpenAIChatCompletionsModel(
        model=cfg.AGENT_MODEL,
        openai_client=sf_async_client,
    ),
)


# ── 流式查询 ──────────────────────────────────────────────────────────

async def query_agent(prompt: str) -> str:
    """执行一次 Agent 查询，流式输出到终端。"""
    streamed_run = Runner.run_streamed(agent, prompt, max_turns=10)

    async for event in streamed_run.stream_events():
        try:
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
        except Exception as e:
            logger.warning("事件处理异常: %s", e)

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
