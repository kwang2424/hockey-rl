"""Scoring a policy against fixed anchors, in log-odds of goal share.

Why this exists
---------------

`train.py` used to pick `best.pt` by goal difference per minute against
ChaseBot. Over a 100M-step run that metric read roughly -9/min at 10M steps and
roughly -7/min at 70M, wandering non-monotonically in between, while a
head-to-head ladder over the same snapshots showed goal share climbing 0.107 ->
0.579. The selection signal was nearly blind to a 5x change in skill, so
`best.pt` was in practice an arbitrary snapshot of the run.

Two separate things were wrong, and both are fixed here.

**The statistic.** Goal difference per minute is bounded by how fast the
opponent scores, so it compresses hard once you are losing badly (or winning
easily) and most of its range is spent on differences you do not care about.
Log-odds of goal share has no such ceiling: the same ChaseBot games, rescored
as ``logit(share)``, move -3.9 -> -1.6 monotonically across the run.

**The anchor set.** One opponent only has resolution near its own strength.
Against ChaseBot the learner was pinned near zero goals for the first 30M
steps; against RandomBot it had saturated by 40M. Neither covers the range
alone, and *between* them they do -- random carries the early signal, chase
carries the late one. So score against several anchors spanning the range and
average.

The average of log-odds is the natural combination: under Bradley-Terry it is
(up to an additive constant from the anchors' own strengths) the maximum
likelihood estimate of the learner's rating, and since the anchors are fixed
checkpoints that constant is the same for every run scored this way. Ratings
are therefore comparable across runs, which goal-diff-per-min never was.
"""

import math


def share_logit(goals_for, goals_against):
    """Log-odds of goal share, Laplace-smoothed.

    Smoothing rather than clipping to a floor: a shutout should not produce an
    infinite rating, but it should still count as more extreme when it happens
    over 200 goals than over 5. ``(gf + 1) / (gf + ga + 2)`` does that on its
    own and leaves no arbitrary epsilon to tune.
    """
    gf, ga = int(goals_for), int(goals_against)
    if gf < 0 or ga < 0:
        raise ValueError(f"goal counts must be non-negative, got {gf}, {ga}")
    share = (gf + 1.0) / (gf + ga + 2.0)
    return math.log(share / (1.0 - share))


def rating(per_anchor):
    """Mean log-odds across anchors. ``per_anchor``: {name: (gf, ga)}.

    Equal weights on purpose. A saturated anchor contributes a roughly constant
    term, so it neither drowns out nor distorts the anchors that still have
    resolution -- it just stops moving, and the informative ones carry the
    score.
    """
    if not per_anchor:
        raise ValueError("rating needs at least one anchor")
    return sum(share_logit(gf, ga) for gf, ga in per_anchor.values()) / len(per_anchor)
