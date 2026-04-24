"""Document auto-classifier — scores structure + length to decide short vs long storage."""
import re

from src import config as cfg
from src.ingester import IngestResult


class DocumentClassifier:
    """Determines whether a document is 'short' (key-value) or 'long' (PageIndex tree)."""

    def classify(self, result: IngestResult) -> str:
        """Return 'short' or 'long' based on weighted structure + length scores."""
        structure_score = self._structure_score(result)
        length_score = self._length_score(result)
        final = (
            structure_score * cfg.CLASSIFIER_STRUCTURE_WEIGHT
            + length_score * cfg.CLASSIFIER_LENGTH_WEIGHT
        )
        return "long" if final > 0.5 else "short"

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (character mode, zero dependencies).

        Chinese chars ~ 1.5 tokens, English words ~ 1.3 tokens,
        non-Chinese alphanumeric chars ~ 0.25 tokens (sub-word units).
        """
        if not text:
            return 0
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        non_chinese = re.sub(r"[\u4e00-\u9fff]", " ", text)
        english_words = len(non_chinese.split())
        # Count non-Chinese alphanumeric characters for continuous strings.
        non_cjk_chars = len(re.findall(r"[a-zA-Z0-9]", text))
        non_cjk_excess = max(0, non_cjk_chars - english_words * 5)
        return int(chinese_chars * 1.5 + english_words * 1.3 + non_cjk_excess * 0.25)

    def _structure_score(self, result: IngestResult) -> float:
        """Structure score 0-1 based on heading depth and count."""
        if not result.headings:
            return 0.0
        levels = {h.level for h in result.headings}
        depth = max(levels) - min(levels) + 1
        heading_count = len(result.headings)
        score = min(1.0, depth * 0.3 + heading_count * 0.1)
        if result.has_tables:
            score = min(1.0, score + 0.1)
        return score

    def _length_score(self, result: IngestResult) -> float:
        """Length score 0-1 based on token estimate relative to threshold."""
        tokens = self.estimate_tokens(result.text)
        if tokens == 0:
            return 0.0
        return min(1.0, tokens / cfg.CLASSIFIER_TOKEN_THRESHOLD)
