"""Cross-encoder reranker for job relevance scoring.

Uses a cross-encoder model to compute a relevance logit score between a
search query and a job posting. Unlike bi-encoders (semantic_scorer.py),
cross-encoders process the query and document jointly, yielding higher
precision at the cost of speed.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- ~70 MB, CPU-runnable
- Returns raw logits (not probabilities); higher = more relevant
- Typical range: roughly -10 to +10; 0.0 is the neutral point
- Downloaded to ~/.cache/huggingface/ on first call
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from job_research.constants import CROSS_ENCODER_THRESHOLD

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder as _CrossEncoder

CROSS_ENCODER_MODEL: Final[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_CROSS_ENCODER: _CrossEncoder | None = None

__all__ = [
    "CROSS_ENCODER_MODEL",
    "CROSS_ENCODER_THRESHOLD",
    "batch_cross_encode",
    "cross_encode",
    "get_cross_encoder",
]


def get_cross_encoder() -> _CrossEncoder:
    """Return cached CrossEncoder instance (loads on first call).

    The model (~70 MB) is downloaded to ~/.cache/huggingface/ on the
    first invocation and reused for all subsequent calls in the same
    process.
    """
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER = CrossEncoder(CROSS_ENCODER_MODEL)
    return _CROSS_ENCODER


def cross_encode(
    *,
    search_keyword: str,
    job_title: str,
    job_description: str | None = None,
    max_desc_chars: int = 500,
) -> float:
    """Return the cross-encoder logit score for (query, document) pair.

    The ms-marco model returns raw logits, not probabilities. Higher = more
    relevant. Typical range: roughly -10 to +10. 0.0 is the neutral point.
    Returns the raw logit (not clamped) so callers can pick their threshold.

    Args:
        search_keyword: The search intent / role name (query).
        job_title: The job posting title (part of the document).
        job_description: Optional job description text.
        max_desc_chars: Maximum characters from the description to include.

    Returns:
        Raw logit score as a Python float.
    """
    model = get_cross_encoder()

    desc_part = (job_description or "")[:max_desc_chars].strip()
    doc = f"{job_title}. {desc_part}" if desc_part else job_title

    scores = model.predict([(search_keyword, doc)])
    return float(scores[0])


def batch_cross_encode(
    search_keyword: str,
    jobs: list[tuple[str, str | None]],
    max_desc_chars: int = 500,
) -> list[float]:
    """Batch version for efficiency.

    Args:
        search_keyword: The search intent / role name (query).
        jobs: List of (title, description) tuples.
        max_desc_chars: Maximum characters from the description to include.

    Returns:
        List of raw logit scores, one per input job.
    """
    if not jobs:
        return []

    model = get_cross_encoder()

    pairs: list[tuple[str, str]] = []
    for title, description in jobs:
        desc_part = (description or "")[:max_desc_chars].strip()
        doc = f"{title}. {desc_part}" if desc_part else title
        pairs.append((search_keyword, doc))

    scores = model.predict(pairs)
    return [float(s) for s in scores]
