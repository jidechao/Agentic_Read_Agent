import os

import pytest

os.environ.setdefault("SILICONFLOW_API_KEY", "test-key")

from src.classifier import DocumentClassifier
from src.ingester import Heading, IngestResult


@pytest.fixture
def classifier() -> DocumentClassifier:
    return DocumentClassifier()


def test_classify_long_structured_doc(classifier: DocumentClassifier) -> None:
    """Has structure + over threshold -> long."""
    result = IngestResult(
        text="x" * 2000,
        title="长文档",
        headings=[
            Heading(level=1, text="第一章"),
            Heading(level=2, text="1.1"),
            Heading(level=2, text="1.2"),
        ],
        has_tables=True,
    )
    assert classifier.classify(result) == "long"


def test_classify_short_flat_doc(classifier: DocumentClassifier) -> None:
    """No structure + short text -> short."""
    result = IngestResult(text="这是一段短文本。", title="短文档", headings=[])
    assert classifier.classify(result) == "short"


def test_classify_long_by_length_only(classifier: DocumentClassifier) -> None:
    """No clear structure but very long -> long (length weight 0.6 enough)."""
    result = IngestResult(
        text="x" * 5000,
        title="超长无结构",
        headings=[Heading(level=1, text="唯一标题")],
    )
    assert classifier.classify(result) == "long"


def test_estimate_tokens_chinese(classifier: DocumentClassifier) -> None:
    count = classifier.estimate_tokens("这是一段中文文本。")
    assert count > 0
    assert 8 <= count <= 30


def test_estimate_tokens_english(classifier: DocumentClassifier) -> None:
    count = classifier.estimate_tokens("This is an English text for testing.")
    assert count > 0


def test_estimate_tokens_empty(classifier: DocumentClassifier) -> None:
    assert classifier.estimate_tokens("") == 0
