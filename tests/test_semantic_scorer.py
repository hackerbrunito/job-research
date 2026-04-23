"""Tests for the bi-encoder semantic scorer.

These tests load the actual sentence-transformers model the first time they
run (~30s on first run; model is cached in ~/.cache/huggingface/ afterwards).
They are intentionally NOT mocked — the whole point is to verify real
embedding quality for job-title similarity.
"""

from __future__ import annotations

from job_research.semantic_scorer import (
    SEMANTIC_SCORE_THRESHOLD,
    batch_score_relevance,
    score_relevance,
)


class TestScoreRelevance:
    def test_exact_match_scores_high(self) -> None:
        """Same title as keyword should score near 1.0."""
        score = score_relevance(
            search_keyword="Store Development Manager",
            job_title="Store Development Manager",
        )
        assert score >= 0.85, f"Expected >=0.85, got {score:.4f}"

    def test_unrelated_job_scores_below_threshold(self) -> None:
        """Clearly unrelated job should score below threshold."""
        score = score_relevance(
            search_keyword="Store Development Manager",
            job_title="Lawn Operative",
            job_description="Cutting grass and maintaining green areas.",
        )
        assert score < SEMANTIC_SCORE_THRESHOLD, (
            f"Expected <{SEMANTIC_SCORE_THRESHOLD}, got {score:.4f}"
        )

    def test_score_clamped_to_unit_interval(self) -> None:
        """Score must always be in [0, 1]."""
        score = score_relevance(
            search_keyword="Software Engineer",
            job_title="Software Engineer",
        )
        assert 0.0 <= score <= 1.0

    def test_description_scores_high_for_related_role(self) -> None:
        """Both title-only and title+description should score well above threshold."""
        score_no_desc = score_relevance(
            search_keyword="Data Engineer",
            job_title="Data Engineer",
        )
        score_with_desc = score_relevance(
            search_keyword="Data Engineer",
            job_title="Data Engineer",
            job_description="Build and maintain ETL pipelines using Python and Spark.",
        )
        # Adding description dilutes the embedding slightly (longer doc vs short
        # query) but both should still be well above the rejection threshold.
        assert score_no_desc >= SEMANTIC_SCORE_THRESHOLD
        assert score_with_desc >= SEMANTIC_SCORE_THRESHOLD


class TestBatchScoreRelevance:
    def test_returns_same_length_as_input(self) -> None:
        jobs = [("Data Engineer", "pipelines"), ("Lawn Operative", "grass")]
        scores = batch_score_relevance("Data Engineer", jobs)
        assert len(scores) == 2

    def test_all_scores_in_unit_interval(self) -> None:
        jobs = [("Data Engineer", "pipelines"), ("Lawn Operative", "grass")]
        scores = batch_score_relevance("Data Engineer", jobs)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_related_job_scores_higher_than_unrelated(self) -> None:
        scores = batch_score_relevance(
            "Data Engineer",
            [
                ("Data Engineer UK", "Build data pipelines"),
                ("Lawn Operative", "Grass cutting"),
            ],
        )
        assert scores[0] > scores[1], (
            f"Expected data engineer ({scores[0]:.4f}) > lawn operative ({scores[1]:.4f})"
        )

    def test_empty_input_returns_empty_list(self) -> None:
        assert batch_score_relevance("Data Engineer", []) == []

    def test_none_description_handled_gracefully(self) -> None:
        """None description must not raise."""
        scores = batch_score_relevance(
            "Store Development Manager",
            [("Store Manager", None), ("Gardener", None)],
        )
        assert len(scores) == 2
        assert all(0.0 <= s <= 1.0 for s in scores)
