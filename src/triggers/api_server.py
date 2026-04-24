"""FastAPI 触发器。"""
import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

import src.config as cfg
from src.compiler import KnowledgeCompiler
from src.registry import KnowledgeRegistry

logger = logging.getLogger(__name__)
app = FastAPI(title="知识库编译系统 API")


def _get_registry() -> KnowledgeRegistry:
    return KnowledgeRegistry(cfg.KNOWLEDGE_DB)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/compile")
def compile_knowledge(force: bool = False) -> dict:
    registry = _get_registry()
    try:
        compiler = KnowledgeCompiler(registry=registry)
        result = compiler.compile(trigger_type="api", force=force)
        return {
            "status": "completed",
            "processed": result.docs_processed,
            "skipped": result.docs_skipped,
            "deleted": result.docs_deleted,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        registry.close()


@app.get("/api/status")
def status() -> dict:
    registry = _get_registry()
    try:
        docs = registry.list_documents()
        return {
            "total": len(docs),
            "documents": [
                {
                    "id": d["id"],
                    "path": d["source_path"],
                    "status": d["status"],
                    "tier": d["tier"],
                }
                for d in docs
            ],
        }
    finally:
        registry.close()


@app.get("/api/documents")
def list_documents() -> list[dict]:
    registry = _get_registry()
    try:
        docs = registry.list_documents()
        return [
            {
                "id": d["id"],
                "path": d["source_path"],
                "format": d["format"],
                "title": d["title"],
                "status": d["status"],
                "tier": d["tier"],
            }
            for d in docs
        ]
    finally:
        registry.close()


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    dest = cfg.DATA_DIR / file.filename
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"status": "uploaded", "path": str(dest)}


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)
