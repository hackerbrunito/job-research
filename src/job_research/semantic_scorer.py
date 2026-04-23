"""Bi-encoder semantic scorer using sentence-transformers.

Computes a cosine-similarity score (0-1) between a job posting and
a search-intent string. The model is loaded once per process and cached
as a module-level singleton.

Model choice: all-MiniLM-L6-v2
- ~80 MB, runs on CPU
- Good balance of speed and accuracy for short-text similarity
- Downloaded to ~/.cache/huggingface/ on first call
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from job_research.constants import SEMANTIC_SCORE_THRESHOLD

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer

MODEL_NAME: Final[str] = "all-MiniLM-L6-v2"

_MODEL: _SentenceTransformer | None = None

__all__ = [
    "MODEL_NAME",
    "SEMANTIC_SCORE_THRESHOLD",
    "batch_score_relevance",
    "get_scorer",
    "score_relevance",
]


def get_scorer() -> _SentenceTransformer:
    """Return the cached model instance (loads on first call).

    The model (~80 MB) is downloaded to ~/.cache/huggingface/ on the
    first invocation and reused for all subsequent calls in the same
    process.
    """
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def score_relevance(
    *,
    search_keyword: str,
    job_title: str,
    job_description: str | None = None,
    max_desc_chars: int = 500,
) -> float:
    """Return cosine similarity [0, 1] between the search intent and the job.

    Builds two sentences:
      query = search_keyword   (e.g. "Store Development Manager")
      doc   = f"{job_title}. {job_description[:max_desc_chars]}"

    Returns util.cos_sim(query_emb, doc_emb).item() clamped to [0, 1].
    """
    from sentence_transformers import util

    model = get_scorer()

    query = search_keyword
    desc_part = (job_description or "")[:max_desc_chars].strip()
    doc = f"{job_title}. {desc_part}" if desc_part else job_title

    query_emb, doc_emb = model.encode([query, doc], convert_to_tensor=True)
    raw: float = util.cos_sim(query_emb, doc_emb).item()  # type: ignore[union-attr]
    return float(max(0.0, min(1.0, raw)))


def batch_score_relevance(
    search_keyword: str,
    jobs: list[tuple[str, str | None]],
    max_desc_chars: int = 500,
) -> list[float]:
    """Batch version — more efficient when scoring many rows at once.

    Args:
        search_keyword: The search intent / role name.
        jobs: List of (title, description) tuples.
        max_desc_chars: Maximum characters from the description to include.

    Returns:
        List of float scores in [0, 1], one per input job.
    """
    if not jobs:
        return []

    from sentence_transformers import util

    model = get_scorer()

    docs: list[str] = []
    for title, description in jobs:
        desc_part = (description or "")[:max_desc_chars].strip()
        docs.append(f"{title}. {desc_part}" if desc_part else title)

    sentences = [search_keyword, *docs]
    embeddings = model.encode(sentences, convert_to_tensor=True)
    query_emb = embeddings[0]
    doc_embs = embeddings[1:]

    scores: list[float] = []
    for doc_emb in doc_embs:
        raw: float = util.cos_sim(query_emb, doc_emb).item()  # type: ignore[union-attr]
        scores.append(float(max(0.0, min(1.0, raw))))

    return scores
