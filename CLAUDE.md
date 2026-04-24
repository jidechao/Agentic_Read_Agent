# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 运行流程（必须按顺序）

```bash
# 使用虚拟环境
source .venv/bin/activate

python -m src.data_generator              # 1. 生成测试数据 → data/
python -m src.triggers.cli compile        # 2. 编译知识库 → compiled_library/
python -m src.main_agent                  # 3. 启动交互式 Agent
```

编译必须在 Agent 启动之前完成。Agent 启动时检查 `compiled_library/SKILL.md`，缺失则报错退出。

## 环境要求

- Python 3.11+，依赖：`pip install -r requirements.txt`
- `.env` 中必须配置 `SILICONFLOW_API_KEY`（SiliconFlow API 密钥）
- 测试运行：`.venv/bin/python -m pytest tests/ -v`

## 架构

两层导航式 RAG 系统：
- **短文档层**：JSON 键值存储（`short_docs_db.json`），Agent 直接匹配读取
- **长文档层**：PageIndex 树索引 + embedding 聚类，Agent 读 `SKILL.md` → `INDEX.md` → 逐节阅读
- **编译流程**：扫描 → 解析 → 分类（短/长）→ 嵌入（标题+摘要）→ 聚类（轮廓系数自动选 K）→ 原子物化

## 关键注意事项

- **PageIndex 通过 sys.path hack 加载**（`src/config.py` 第 10-13 行），不是 pip 安装的包
- **SiliconFlow 不支持 Responses API**，Agent 使用 `OpenAIChatCompletionsModel`（`src/main_agent.py`）
- **PageIndex 内部使用 LiteLLM**，模型名需要 `openai/` 前缀才能正确路由
- **PageIndex 是 git submodule**（`git clone --recurse-submodules`），有自己的 CLAUDE.md
- **长文档在 INDEX.md 中显示 pageindex_id**，短文档显示 registry id。`get_document_source` 工具支持两种 ID 查询

## 代码风格

- 使用 **ruff** 格式化和 lint：`ruff format src/` 和 `ruff check src/`
- 代码注释和 Agent prompt 使用中文
- Commit message 使用 conventional commits 格式（feat/fix/chore/refactor/docs）
