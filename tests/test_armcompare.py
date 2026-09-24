"""The arm comparison must respect that the run, not the pairing, is the unit."""

import numpy as np
import pytest

from hockey.armcompare import cross_mean, permutation_p


def _matrix(shares):
    """Antisymmetric share matrix from a dict {(i,j): share_of_i}."""
    n = 1 + max(max(k) for k in shares)
    S = np.full((n, n), np.nan)
    for (i, j), v in shares.items():
        S[i][j] = v
        S[j][i] = 1 - v
    return S


def test_cross_mean_uses_only_cross_label_pairs():
    S = _matrix({(0, 1): 0.9, (0, 2): 0.8, (1, 2): 0.4})
    label = np.array([0, 0, 1])
    # only 0-vs-2 and 1-vs-2 count; 0-vs-1 is within arm A
    assert cross_mean(S, label) == pytest.approx((0.8 + 0.4) / 2)


def test_no_effect_gives_a_large_p():
    n = 12
    rng = np.random.default_rng(0)
    S = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            S[i][j] = 0.5
            S[j][i] = 0.5
    label = np.array([0] * 6 + [1] * 6)
    _, p, _ = permutation_p(S, label, 2000, rng)
    assert p > 0.5


def test_a_real_separation_is_detected_at_six_a_side():
    """Arm A beats arm B decisively and consistently."""
    n = 12
    S = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            same = (i < 6) == (j < 6)
            v = 0.5 if same else (0.9 if i < 6 else 0.1)
            S[i][j] = v
            S[j][i] = 1 - v
    label = np.array([0] * 6 + [1] * 6)
    obs, p, _ = permutation_p(S, label, 5000, np.random.default_rng(0))
    assert obs == pytest.approx(0.9)
    assert p < 0.05


def test_two_runs_a_side_cannot_reach_significance():
    """The unit is the run, so four runs cannot produce a significant result
    however lopsided they look. C(4,2)=6 labellings floor the two-sided p at
    1/3 -- which is the guard against reading a two-run experiment as proof."""
    S = np.full((4, 4), np.nan)
    for i in range(4):
        for j in range(i + 1, 4):
            v = 0.5 if (i < 2) == (j < 2) else (1.0 if i < 2 else 0.0)
            S[i][j] = v
            S[j][i] = 1 - v
    label = np.array([0, 0, 1, 1])
    obs, p, _ = permutation_p(S, label, 5000, np.random.default_rng(0))
    assert obs == pytest.approx(1.0)
    assert p > 0.2


def test_missing_pairings_are_skipped_not_counted_as_even():
    """A pairing that scored no goals is absent, not 0.5 -- averaging in a
    phantom draw would drag every effect toward "no difference"."""
    S = _matrix({(0, 1): 0.9, (0, 2): 0.7, (1, 2): 0.5})
    S[0][2] = S[2][0] = np.nan          # that pairing produced no goals
    label = np.array([0, 1, 1])
    assert cross_mean(S, label) == pytest.approx(0.9)
