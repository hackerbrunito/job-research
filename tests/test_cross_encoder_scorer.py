"""Tests for cross_encoder_scorer module.

These tests load the actual cross-encoder model (~70 MB, downloaded on first
run). They verify directional correctness: exact-match queries score above
threshold, unrelated queries score below threshold, and batch ordering is
preserved.
"""

from __future__ import annotations

from job_research.cross_encoder_scorer import (
    CROSS_ENCODER_THRESHOLD,
    batch_cross_encode,
    cross_encode,
)


def test_cross_encode_exact_match_scores_high() -> None:
    """Exact title match should score above the neutral logit threshold."""
    score = cross_encode(
        search_keyword="Store Development Manager",
        job_title="Store Development Manager",
    )
    assert score > CROSS_ENCODER_THRESHOLD


def test_cross_encode_unrelated_scores_low() -> None:
    """Unrelated job should score below the neutral logit threshold."""
    score = cross_encode(
        search_keyword="Store Development Manager",
        job_title="Lawn Operative",
        job_description="Maintaining green areas.",
    )
    assert score < CROSS_ENCODER_THRESHOLD


def test_batch_cross_encode_length() -> None:
    """Batch output length must match input length."""
    scores = batch_cross_encode(
        "Data Engineer",
        [("Data Engineer", "pipelines"), ("Lawn Operative", "grass")],
    )
    assert len(scores) == 2


def test_batch_orders_correctly() -> None:
    """Relevant job should score higher than unrelated job in the same batch."""
    scores = batch_cross_encode(
        "Data Engineer",
        [("Lawn Operative", "grass"), ("Data Engineer", "pipelines")],
    )
    assert scores[1] > scores[0]


def test_batch_empty_returns_empty() -> None:
    """Empty job list should return an empty list without error."""
    scores = batch_cross_encode("Data Engineer", [])
    assert scores == []


def test_cross_encode_description_truncation() -> None:
    """Very long descriptions should be truncated without raising."""
    long_desc = "data pipelines " * 300  # well over 500 chars
    score = cross_encode(
        search_keyword="Data Engineer",
        job_title="Data Engineer",
        job_description=long_desc,
        max_desc_chars=500,
    )
    assert isinstance(score, float)


def test_cross_encode_no_description() -> None:
    """Omitting description should work without error."""
    score = cross_encode(
        search_keyword="Software Engineer",
        job_title="Software Engineer",
    )
    assert isinstance(score, float)


def test_singleton_is_reused() -> None:
    """Two calls should return the same CrossEncoder object."""
    from job_research.cross_encoder_scorer import get_cross_encoder

    m1 = get_cross_encoder()
    m2 = get_cross_encoder()
    assert m1 is m2
