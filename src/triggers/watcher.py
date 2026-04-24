"""文件监控触发器 — 使用 watchdog 库。"""
import logging
import time
from pathlib import Path
from threading import Timer

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import src.config as cfg
from src.compiler import KnowledgeCompiler
from src.registry import KnowledgeRegistry

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".pdf", ".md", ".markdown", ".docx", ".doc", ".html", ".htm"}
DEBOUNCE_SECONDS = 0.5


class CompilationHandler(FileSystemEventHandler):
    def __init__(self) -> None:
        self._timer: Timer | None = None
        self._dirty = False

    def on_any_event(self, event) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in SUPPORTED_EXTS:
            return
        logger.info("检测到文件变化: %s", path.name)
        self._dirty = True
        self._schedule_compile()

    def _schedule_compile(self) -> None:
        if self._timer:
            self._timer.cancel()
        self._timer = Timer(DEBOUNCE_SECONDS, self._do_compile)
        self._timer.daemon = True
        self._timer.start()

    def _do_compile(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        try:
            registry = KnowledgeRegistry(cfg.KNOWLEDGE_DB)
            try:
                compiler = KnowledgeCompiler(registry=registry)
                result = compiler.compile(trigger_type="watcher")
                logger.info("自动编译完成: 处理 %d, 跳过 %d", result.docs_processed, result.docs_skipped)
            finally:
                registry.close()
        except Exception as e:
            logger.error("自动编译失败: %s", e)


def run_watcher(watch_dir: str | None = None) -> None:
    dir_path = Path(watch_dir) if watch_dir else cfg.DATA_DIR
    handler = CompilationHandler()
    observer = Observer()
    observer.schedule(handler, str(dir_path), recursive=True)
    observer.start()
    logger.info("文件监控已启动: %s", dir_path)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
