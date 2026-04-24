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
