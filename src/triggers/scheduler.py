"""定时调度触发器 — 使用 APScheduler。"""
import logging

from apscheduler.schedulers.blocking import BlockingScheduler

import src.config as cfg
from src.compiler import KnowledgeCompiler
from src.registry import KnowledgeRegistry

logger = logging.getLogger(__name__)


def _compile_job() -> None:
    registry = KnowledgeRegistry(cfg.KNOWLEDGE_DB)
    try:
        compiler = KnowledgeCompiler(registry=registry)
        result = compiler.compile(trigger_type="scheduler")
        logger.info("定时编译完成: 处理 %d, 跳过 %d", result.docs_processed, result.docs_skipped)
    except Exception as e:
        logger.error("定时编译失败: %s", e)
    finally:
        registry.close()


def run_scheduler(cron: str = "0 2 * * *") -> None:
    """Run scheduler with given cron expression. Format: minute hour day month day_of_week"""
    parts = cron.split()
    if len(parts) != 5:
        raise ValueError(f"Invalid cron expression: {cron}")

    scheduler = BlockingScheduler()
    scheduler.add_job(
        _compile_job,
        "cron",
        minute=parts[0],
        hour=parts[1],
        day=parts[2],
        month=parts[3],
        day_of_week=parts[4],
    )
    logger.info("定时调度已启动: cron=%s", cron)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
