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

    # LiteLLM 需要 openai/ 前缀才能正确路由到自定义 base_url
    litellm_model = f"openai/{cfg.PAGEINDEX_MODEL}"
    client = PageIndexClient(
        workspace=cfg.PAGEINDEX_WORKSPACE,
        model=litellm_model,
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
    labels = kmeans.fit_predict(np.array(vectors))
    unique, counts = np.unique(labels, return_counts=True)
    logger.info("聚类完成，标签分布: %s", dict(zip(unique.tolist(), counts.tolist())))

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
        "以下是同一分类的企业知识文档标题列表：\n"
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
        return "cluster-default", "默认分类"


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
