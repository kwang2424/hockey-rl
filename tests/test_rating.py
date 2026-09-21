"""The selection metric. See hockey/rating.py for why it replaced goal diff."""

import math

import pytest

from hockey.rating import rating, share_logit
from train import anchor_key, eval_fields


def test_even_result_is_zero():
    assert share_logit(50, 50) == 0.0


def test_monotonic_in_goal_share():
    prev = -math.inf
    for gf in range(0, 200, 10):
        cur = share_logit(gf, 200 - gf)
        assert cur > prev
        prev = cur


def test_shutouts_are_finite_and_scale_with_sample_size():
    """Laplace smoothing, not a clip floor: a bigger shutout counts as more
    extreme, and nothing is ever infinite."""
    small, big = share_logit(0, 5), share_logit(0, 500)
    assert math.isfinite(small) and math.isfinite(big)
    assert big < small


def test_resolves_where_goal_difference_saturates():
    """The actual failure. Against ChaseBot the learner went 3-227 early and
    25-185 late; goal difference per minute barely moved across that change
    while the ladder showed skill roughly quintupling."""
    early, late = share_logit(3, 227), share_logit(25, 185)
    assert late - early > 1.5


def test_a_saturated_anchor_does_not_pin_the_rating():
    """A hopeless anchor contributes a near-constant term, so an improvement
    measured by a second anchor still moves the score."""
    before = rating({"hard": (2, 300), "easy": (100, 40)})
    after = rating({"hard": (2, 300), "easy": (160, 20)})
    assert after > before


def test_rating_rejects_an_empty_anchor_set():
    with pytest.raises(ValueError):
        rating({})


def test_negative_counts_are_rejected():
    with pytest.raises(ValueError):
        share_logit(-1, 5)


def test_anchor_keys_are_csv_safe_and_distinct():
    keys = [anchor_key(s) for s in
            ("chase", "random", "checkpoints/v2-control-gated-reward.pt",
             "checkpoints/v6-100m-compute.pt")]
    assert keys == ["chase", "random", "v2_control_gated_reward", "v6_100m_compute"]
    assert len(set(keys)) == len(keys)


def test_schema_declares_a_column_per_anchor_plus_the_rating():
    fields = eval_fields(("chase", "checkpoints/v2-control-gated-reward.pt"))
    assert "eval_rating" in fields
    for key in ("chase", "v2_control_gated_reward"):
        assert f"vs_{key}_goals_for" in fields
        assert f"vs_{key}_goals_against" in fields
    assert len(fields) == 2 * 4 + 1
