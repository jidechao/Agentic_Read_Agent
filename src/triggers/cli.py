"""手动 CLI 触发器。"""
import argparse
import logging
import sys
from pathlib import Path

import src.config as cfg
from src.compiler import KnowledgeCompiler
from src.registry import KnowledgeRegistry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src", description="知识库编译系统")
    sub = parser.add_subparsers(dest="command")

    # compile
    p_compile = sub.add_parser("compile", help="编译知识库")
    p_compile.add_argument("--force", action="store_true", help="强制全量重编译")
    p_compile.add_argument("--data-dir", type=str, help="源文档目录（默认 data/）")
    p_compile.add_argument("--verbose", "-v", action="store_true")

    # status
    p_status = sub.add_parser("status", help="查看注册表状态")

    # watch
    p_watch = sub.add_parser("watch", help="监控文件变化并自动编译")
    p_watch.add_argument("--dir", type=str, help="监控目录")

    # schedule
    p_schedule = sub.add_parser("schedule", help="定时编译")
    p_schedule.add_argument("--cron", type=str, default="0 2 * * *", help="Cron 表达式")

    # serve
    p_serve = sub.add_parser("serve", help="启动 API 服务")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--host", type=str, default="0.0.0.0")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "compile":
        data_dir = Path(args.data_dir) if args.data_dir else None
        registry = KnowledgeRegistry(cfg.KNOWLEDGE_DB)
        try:
            compiler = KnowledgeCompiler(registry=registry)
            result = compiler.compile(
                trigger_type="manual", force=args.force, data_dir=data_dir
            )
            print(f"编译完成: 处理 {result.docs_processed}, 跳过 {result.docs_skipped}, 删除 {result.docs_deleted}")
        finally:
            registry.close()

    elif args.command == "status":
        registry = KnowledgeRegistry(cfg.KNOWLEDGE_DB)
        try:
            docs = registry.list_documents()
            if not docs:
                print("注册表为空")
            else:
                for doc in docs:
                    print(f"  {doc['source_path']:40s} {doc['status']:12s} {doc['tier']:8s} {doc.get('title', '')}")
        finally:
            registry.close()

    elif args.command == "watch":
        from src.triggers.watcher import run_watcher
        run_watcher(getattr(args, "dir", None))

    elif args.command == "schedule":
        from src.triggers.scheduler import run_scheduler
        run_scheduler(args.cron)

    elif args.command == "serve":
        from src.triggers.api_server import run_server
        run_server(host=args.host, port=args.port)

    return 0
