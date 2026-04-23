"""SetFit few-shot classifier for per-profile job-title relevance scoring.

Fine-tunes a sentence-transformer on the user's triage labels (accept/reject)
using contrastive learning, then uses a logistic-regression head to produce
a relevance probability for new job titles.

Models are trained in-memory and cached per profile_id. They are NOT persisted
to disk in this wave — re-train after a process restart by calling
train_for_profile() again.

Base model: sentence-transformers/paraphrase-MiniLM-L3-v2
  - ~18 MB, runs on CPU, fast to fine-tune
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final

from job_research.constants import SETFIT_MIN_EXAMPLES_PER_CLASS, SETFIT_SCORE_THRESHOLD

if TYPE_CHECKING:
    from setfit import SetFitModel

# ---- public constants -------------------------------------------------------

SETFIT_BASE_MODEL: Final[str] = "sentence-transformers/paraphrase-MiniLM-L3-v2"

__all__ = [
    "SETFIT_BASE_MODEL",
    "SETFIT_MIN_EXAMPLES_PER_CLASS",
    "SETFIT_SCORE_THRESHOLD",
    "get_training_summary",
    "is_trained",
    "predict",
    "train_for_profile",
]

# ---- module-level model cache -----------------------------------------------

# Per-profile model cache: {profile_id: trained SetFitModel}
_MODELS: dict[str, SetFitModel] = {}

_log = logging.getLogger(__name__)

# ---- label mapping ----------------------------------------------------------

_LABEL_TO_INT: Final[dict[str, int]] = {"accept": 1, "reject": 0}


# ---- public API -------------------------------------------------------------


def get_training_summary(
    profile_id: str,
    labels: list[dict[str, Any]],
) -> dict[str, int]:
    """Return counts of accept/reject/unsure labels plus a total.

    Args:
        profile_id: Profile identifier (not used in counting, kept for
            consistent call signature with other profile-scoped helpers).
        labels: List of label dicts from ``list_title_labels()``
            — each must have a ``"label"`` key ∈ {'accept','reject','unsure'}.

    Returns:
        ``{"accept": N, "reject": N, "unsure": N, "total": N}``
    """
    counts: dict[str, int] = {"accept": 0, "reject": 0, "unsure": 0}
    for row in labels:
        lbl = row.get("label", "")
        if lbl in counts:
            counts[lbl] += 1
    return {**counts, "total": sum(counts.values())}


def is_trained(profile_id: str) -> bool:
    """Return True if a trained model exists in the in-memory cache."""
    return profile_id in _MODELS


def train_for_profile(
    profile_id: str,
    labels: list[dict[str, Any]],
) -> bool:
    """Train (or retrain) the SetFit model for a profile.

    Converts triage labels to binary training data:
      'accept'  → 1  (relevant)
      'reject'  → 0  (not relevant)
      'unsure'  → skipped

    Training is skipped and ``False`` is returned if either class has fewer
    than ``SETFIT_MIN_EXAMPLES_PER_CLASS`` examples.

    On success the trained model is stored in ``_MODELS[profile_id]`` and
    ``True`` is returned.

    Args:
        profile_id: Identifier used as cache key.
        labels: List of label dicts from ``list_title_labels()``
            — each must have ``"title_norm"`` and ``"label"`` keys.

    Returns:
        ``True`` if training succeeded, ``False`` if there were not enough
        labelled examples.
    """
    # Build binary training data (skip 'unsure')
    texts: list[str] = []
    int_labels: list[int] = []
    for row in labels:
        lbl = row.get("label", "")
        if lbl not in _LABEL_TO_INT:
            continue
        title = row.get("title_norm", "")
        if not title:
            continue
        texts.append(title)
        int_labels.append(_LABEL_TO_INT[lbl])

    # Count per-class examples
    n_accept = int_labels.count(1)
    n_reject = int_labels.count(0)

    if (
        n_accept < SETFIT_MIN_EXAMPLES_PER_CLASS
        or n_reject < SETFIT_MIN_EXAMPLES_PER_CLASS
    ):
        _log.warning(
            "setfit.skip_training: not enough examples "
            "(accept=%d, reject=%d, min_per_class=%d)",
            n_accept,
            n_reject,
            SETFIT_MIN_EXAMPLES_PER_CLASS,
        )
        return False

    # Lazy imports — setfit/torch are large; only pull in when training
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    train_dataset = Dataset.from_dict({"text": texts, "label": int_labels})

    model = SetFitModel.from_pretrained(SETFIT_BASE_MODEL)

    args = TrainingArguments(
        batch_size=16,
        num_epochs=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )
    trainer.train()

    _MODELS[profile_id] = model
    _log.info(
        "setfit.trained profile_id=%s accept=%d reject=%d",
        profile_id,
        n_accept,
        n_reject,
    )
    return True


def predict(
    profile_id: str,
    texts: list[str],
) -> list[float]:
    """Return relevance probability scores in [0, 1] for each text.

    If no model is trained for this profile every text gets ``1.0``
    (assume relevant — avoids false negatives when no training data exists).

    Args:
        profile_id: Profile whose model to use.
        texts: Job titles (or any short strings) to score.

    Returns:
        List of floats in [0, 1], one per input text.
    """
    if not texts:
        return []

    model = _MODELS.get(profile_id)
    if model is None:
        return [1.0] * len(texts)

    # predict_proba returns shape (n_texts, n_classes); column 1 is P(relevant)
    import numpy as np

    proba = model.predict_proba(texts, as_numpy=True)  # type: ignore[arg-type]
    # proba shape: (n, 2) — column 0 = P(reject), column 1 = P(accept)
    if isinstance(proba, np.ndarray) and proba.ndim == 2:
        scores = proba[:, 1].tolist()
    else:
        # Fallback: 1-D array of class predictions — convert to float
        scores = [float(p) for p in proba]

    return [float(s) for s in scores]
