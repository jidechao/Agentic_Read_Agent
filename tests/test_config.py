"""验证配置模块的新增项存在且可访问。"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SILICONFLOW_API_KEY", "test-key-for-config-test")


def test_config_has_classifier_settings():
    import src.config as cfg

    assert hasattr(cfg, "CLASSIFIER_TOKEN_THRESHOLD")
    assert cfg.CLASSIFIER_TOKEN_THRESHOLD > 0
    assert hasattr(cfg, "CLASSIFIER_STRUCTURE_WEIGHT")
    assert 0 <= cfg.CLASSIFIER_STRUCTURE_WEIGHT <= 1
    assert hasattr(cfg, "CLASSIFIER_LENGTH_WEIGHT")
    assert 0 <= cfg.CLASSIFIER_LENGTH_WEIGHT <= 1
    assert hasattr(cfg, "CLASSIFIER_USE_EXACT_TOKENS")
    assert isinstance(cfg.CLASSIFIER_USE_EXACT_TOKENS, bool)


def test_config_has_registry_path():
    import src.config as cfg

    assert hasattr(cfg, "KNOWLEDGE_DB")
    assert isinstance(cfg.KNOWLEDGE_DB, Path)


def test_config_has_data_dir():
    import src.config as cfg

    assert hasattr(cfg, "DATA_DIR")
    assert isinstance(cfg.DATA_DIR, Path)
