"""Tests for the SetFit few-shot classifier module.

The end-to-end train+predict test (~30-60 s on CPU) is included because
SetFit trains in seconds on a tiny dataset and the result is deterministic
enough to assert relative score ordering.
"""

from __future__ import annotations

import pytest

from job_research.setfit_classifier import (
    get_training_summary,
    is_trained,
    predict,
    train_for_profile,
)

# --------------------------------------------------------------------------- #
# Fast unit tests (no training required)
# --------------------------------------------------------------------------- #


def test_predict_returns_1_when_no_model() -> None:
    """No trained model → all texts get score 1.0 (safe default)."""
    scores = predict("nonexistent-profile", ["Store Manager", "Lawn Operative"])
    assert scores == [1.0, 1.0]


def test_predict_empty_texts_returns_empty() -> None:
    scores = predict("nonexistent-profile", [])
    assert scores == []


def test_get_training_summary_counts() -> None:
    labels = [
        {"title_norm": "store development manager", "label": "accept"},
        {"title_norm": "retail manager", "label": "accept"},
        {"title_norm": "lawn operative", "label": "reject"},
        {"title_norm": "kitchen supervisor", "label": "reject"},
        {"title_norm": "unclear", "label": "unsure"},
    ]
    summary = get_training_summary("p", labels)
    assert summary == {"accept": 2, "reject": 2, "unsure": 1, "total": 5}


def test_get_training_summary_empty() -> None:
    assert get_training_summary("p", []) == {
        "accept": 0,
        "reject": 0,
        "unsure": 0,
        "total": 0,
    }


def test_is_trained_false_before_training() -> None:
    assert not is_trained("brand-new-profile-xyz")


def test_train_returns_false_when_not_enough_labels() -> None:
    """Fewer than SETFIT_MIN_EXAMPLES_PER_CLASS per class → training skipped."""
    labels = [
        {"title_norm": "store manager", "label": "accept"},
        {"title_norm": "lawn operative", "label": "reject"},
    ]  # only 1 per class, need 4
    result = train_for_profile("test-profile-labels", labels)
    assert result is False


def test_train_returns_false_when_only_one_class() -> None:
    """Only accept labels → reject count is 0 → training skipped."""
    labels = [{"title_norm": f"role {i}", "label": "accept"} for i in range(8)]
    result = train_for_profile("test-profile-one-class", labels)
    assert result is False


def test_train_skips_unsure_labels() -> None:
    """Unsure labels should be ignored; if remaining are too few → False."""
    labels = [
        {"title_norm": "store manager", "label": "accept"},
        {"title_norm": "lawn operative", "label": "reject"},
        {"title_norm": "vague role", "label": "unsure"},
        {"title_norm": "another vague", "label": "unsure"},
    ]
    result = train_for_profile("test-profile-skip-unsure", labels)
    assert result is False


# --------------------------------------------------------------------------- #
# End-to-end test (involves actual model download + training, ~30-60s on CPU)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_train_and_predict() -> None:
    """End-to-end: train on a tiny set, predict on new texts."""
    labels = [
        {"title_norm": "store development manager", "label": "accept"},
        {"title_norm": "head of store expansion", "label": "accept"},
        {"title_norm": "retail real estate manager", "label": "accept"},
        {"title_norm": "store openings director", "label": "accept"},
        {"title_norm": "lawn operative", "label": "reject"},
        {"title_norm": "kitchen supervisor", "label": "reject"},
        {"title_norm": "restaurant manager", "label": "reject"},
        {"title_norm": "accounts manager", "label": "reject"},
    ]
    trained = train_for_profile("wave7d-test", labels)
    assert trained is True
    assert is_trained("wave7d-test")

    scores = predict("wave7d-test", ["Store Development Manager UK", "Lawn Operative"])
    assert len(scores) == 2
    # All scores must be valid probabilities
    for s in scores:
        assert 0.0 <= s <= 1.0
    # Store Development Manager should score higher than Lawn Operative
    assert scores[0] > scores[1]
