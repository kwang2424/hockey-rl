#!/usr/bin/env python3
"""Evaluation harness: numbers, not vibes.

    python -m hockey.evaluate --a runs/v0/best.pt --b chase
    python -m hockey.evaluate --a runs/v0/best.pt --b runs/v0/latest.pt --games 400

Reports goals, shots, possession and an approximate confidence interval on the
goal rate, because a 64-env eval can easily swing by a couple of goals and
look like progress.
"""

import argparse
import json

import numpy as np

from .config import DEFAULT
from .rollout import play


def evaluate(spec_a, spec_b, envs=128, steps=600, seed=0, deterministic=True, swap=True):
    """Play A vs B, then optionally swap ends and play again.

    Swapping matters: it cancels any residual side advantage, so what you are
    left with is a comparison of the policies rather than of the two ends.
    """
    from .watch import resolve_policy
    pa, na = resolve_policy(spec_a)
    pb, nb = resolve_policy(spec_b)

    s1, _ = play(pa, pb, num_envs=envs, steps=steps, seed=seed, deterministic=deterministic)
    gf, ga = s1["goals_a"], s1["goals_b"]
    shots_f, shots_a = s1["shots_a"], s1["shots_b"]
    poss = [s1["possession_a_frac"]]
    minutes = s1["sim_minutes"]

    if swap:
        s2, _ = play(pb, pa, num_envs=envs, steps=steps, seed=seed + 1,
                     deterministic=deterministic)
        gf += s2["goals_b"]; ga += s2["goals_a"]
        shots_f += s2["shots_b"]; shots_a += s2["shots_a"]
        poss.append(1.0 - s2["possession_a_frac"])
        minutes += s2["sim_minutes"]

    total = gf + ga
    share = gf / total if total else 0.5
    # Binomial standard error on "what fraction of goals were A's".
    se = np.sqrt(share * (1 - share) / total) if total else float("nan")
    return {
        "a": na, "b": nb,
        "goals_for": int(gf), "goals_against": int(ga),
        "goal_share": round(float(share), 4),
        "goal_share_95ci": [round(float(max(0, share - 1.96 * se)), 4),
                            round(float(min(1, share + 1.96 * se)), 4)],
        "goal_diff_per_min": round((gf - ga) / minutes, 3),
        "shots_for": int(shots_f), "shots_against": int(shots_a),
        "possession_frac": round(float(np.mean(poss)), 4),
        "sim_minutes": round(minutes, 1),
        "decisive": bool(total > 0 and (share - 1.96 * se > 0.5 or share + 1.96 * se < 0.5)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="policy A: path.pt | chase | random | still")
    ap.add_argument("--b", default="chase", help="policy B")
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-swap", action="store_true")
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()

    out = evaluate(args.a, args.b, args.envs, args.steps, args.seed,
                   deterministic=not args.stochastic, swap=not args.no_swap)
    print(json.dumps(out, indent=2))
    verdict = ("A is better" if out["goal_share"] > 0.5 else "B is better") if out["decisive"] \
        else "too close to call (widen --envs/--steps)"
    print(f"\n{out['a']}  vs  {out['b']}: {out['goals_for']}-{out['goals_against']} "
          f"over {out['sim_minutes']:.0f} sim-minutes -> {verdict}")


if __name__ == "__main__":
    main()
