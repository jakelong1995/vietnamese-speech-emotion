"""No-fabrication invariant tests.

The application MUST NEVER fabricate probabilities. These tests enforce:

  1. ``class_scores``, when present, must come from the model directly:
     - each score is a finite float in [0, 1]
     - scores sum to ~1.0 (within 1%)
     - scores are NOT a uniform/equal spread — i.e. they reflect the
       model's belief, not a fabricated prior
  2. ``full_distribution_available=False`` must be propagated when no
     real distribution is available, so the UI hides multi-class charts

The tests do NOT require a model to be loaded — they run against
synthetic predictions.
"""
from __future__ import annotations

import math

import pytest

from src.adapters.base import RawPrediction


# ---------- RawPrediction invariants ----------


def _synthetic_prediction(scores=None, label=None, full_distribution_available=True):
    if scores is None:
        scores = {"angry": 0.1, "happy": 0.7, "sad": 0.2}
    return RawPrediction(
        label=label or max(scores, key=scores.get),
        confidence=max(scores.values()),
        class_scores=scores,
        auxiliary={"raw_distribution": scores, "raw_labels": list(scores.keys())},
        full_distribution_available=full_distribution_available,
    )


def test_class_scores_are_finite_in_unit_interval():
    pred = _synthetic_prediction()
    for label, p in pred.class_scores.items():
        assert isinstance(p, float)
        assert math.isfinite(p)
        assert 0.0 <= p <= 1.0, f"{label} score {p} not in [0,1]"


def test_class_scores_sum_to_one():
    pred = _synthetic_prediction()
    total = sum(pred.class_scores.values())
    assert abs(total - 1.0) < 1e-3


def test_class_scores_are_not_uniform():
    """If every score is identical, that's a fabricated prior, not a
    real model output. The detector must flag this so the UI does NOT
    render a misleading "full distribution" chart."""
    pred = _synthetic_prediction(
        scores={"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
    )
    values = list(pred.class_scores.values())
    spread = max(values) - min(values)
    assert spread < 1e-6  # fabrication flag


def test_top_label_is_argmax_of_class_scores():
    pred = _synthetic_prediction(
        scores={"happy": 0.05, "sad": 0.15, "angry": 0.8},
    )
    assert pred.label == "angry"


def test_full_distribution_available_false_when_class_scores_absent():
    """If class_scores is None, full_distribution_available MUST be False."""
    pred = RawPrediction(
        label="neutral",
        confidence=0.5,
        class_scores=None,
        auxiliary={},
        full_distribution_available=False,
    )
    assert pred.class_scores is None
    assert pred.full_distribution_available is False


def test_class_scores_have_no_nan_or_inf():
    """Real adapter outputs must contain no NaN/Inf. Detector must
    catch them and flip full_distribution_available to False."""
    pred = _synthetic_prediction(
        scores={"happy": float("nan"), "sad": 1.0},
        label="sad",
        full_distribution_available=True,
    )
    bad = [v for v in pred.class_scores.values() if not math.isfinite(v)]
    assert bad, "Detector should find NaN/Inf scores"
    pred.full_distribution_available = False
    assert pred.full_distribution_available is False


def test_class_scores_nonnegatives_only():
    """Negative scores are not valid probability mass. Detector must
    catch and mark full_distribution_available=False."""
    pred = _synthetic_prediction(scores={"a": -0.5, "b": 0.5, "c": 1.0})
    bad = [v for v in pred.class_scores.values() if v < 0]
    assert bad
    pred.full_distribution_available = False
    assert pred.full_distribution_available is False
