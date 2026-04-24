import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("SILICONFLOW_API_KEY", "test-key")


def test_cli_compile_no_args():
    """No command shows help, returns 1."""
    from src.triggers.cli import main

    assert main([]) == 1


def test_cli_status_empty(tmp_path, monkeypatch):
    """Status command with empty registry shows empty message."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.config.KNOWLEDGE_DB", db_path)
    from src.triggers.cli import main

    assert main(["status"]) == 0


def test_api_health():
    """Health endpoint returns ok."""
    from src.triggers.api_server import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_api_status_empty(tmp_path, monkeypatch):
    """Status endpoint with empty registry."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.config.KNOWLEDGE_DB", db_path)
    from src.triggers.api_server import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.get("/api/status")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


def test_api_upload(tmp_path, monkeypatch):
    """Upload endpoint saves file."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr("src.config.DATA_DIR", data_dir)
    from src.triggers.api_server import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.post(
        "/api/documents/upload", files={"file": ("test.md", b"# Hello")}
    )
    assert resp.status_code == 200
    assert (data_dir / "test.md").exists()


def test_api_list_documents_empty(tmp_path, monkeypatch):
    """Documents endpoint with empty registry returns empty list."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.config.KNOWLEDGE_DB", db_path)
    from src.triggers.api_server import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.get("/api/documents")
    assert resp.status_code == 200
    assert resp.json() == []


def test_scheduler_invalid_cron():
    """Invalid cron expression raises ValueError."""
    from src.triggers.scheduler import run_scheduler

    with pytest.raises(ValueError, match="Invalid cron expression"):
        run_scheduler("bad cron")


def test_watcher_handler_ignores_dirs(tmp_path):
    """CompilationHandler ignores directory events."""
    from src.triggers.watcher import CompilationHandler

    handler = CompilationHandler()

    class FakeEvent:
        is_directory = True
        src_path = str(tmp_path / "subdir")

    handler.on_any_event(FakeEvent())
    assert not handler._dirty


def test_watcher_handler_ignores_unsupported_ext(tmp_path):
    """CompilationHandler ignores unsupported file extensions."""
    from src.triggers.watcher import CompilationHandler

    handler = CompilationHandler()

    class FakeEvent:
        is_directory = False
        src_path = str(tmp_path / "image.png")

    handler.on_any_event(FakeEvent())
    assert not handler._dirty


def test_watcher_handler_marks_dirty_for_supported(tmp_path):
    """CompilationHandler sets dirty flag for supported file types."""
    from src.triggers.watcher import CompilationHandler

    handler = CompilationHandler()

    class FakeEvent:
        is_directory = False
        src_path = str(tmp_path / "notes.md")

    handler.on_any_event(FakeEvent())
    assert handler._dirty
    # Clean up the timer
    if handler._timer:
        handler._timer.cancel()
