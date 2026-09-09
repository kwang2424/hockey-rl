#!/usr/bin/env python3
"""Round-robin head-to-head between checkpoints.

    python -m hockey.ladder                       # everything in checkpoints/ + baselines
    python -m hockey.ladder --add runs/v6/best.pt # include a candidate
    python -m hockey.ladder --quick               # fewer envs, rough ranking

Why this exists, stated plainly: head-to-head play is the only measurement in
this project that ever contradicted a confident conclusion. Behavioural
metrics, return curves and per-step reward arithmetic all agreed with each
other and were wrong together -- three stacked reward changes each verified to
do what it was designed to do produced a policy that lost 10-56 to the
checkpoint they were meant to improve on.

So: keep old checkpoints, and make playing them against each other a single
command rather than something you get around to.

Every pair plays both ends, so a side advantage cannot masquerade as skill.
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np

from .config import DEFAULT
from .rollout import play


def _name(spec):
    """A label you can tell apart in a table.

    Checkpoints inside runs/ are all called best.pt or latest.pt, so the bare
    basename turns every candidate into "best" -- which is exactly as useful
    as it sounds when three of them are in the same ladder.
    """
    if spec in ("chase", "random", "still"):
        return spec
    stem = os.path.splitext(os.path.basename(spec))[0]
    if stem in ("best", "latest", "final"):
        parent = os.path.basename(os.path.dirname(spec))
        return f"{parent}/{stem}" if parent else stem
    return stem


def run_ladder(specs, envs=96, steps=400, seed=0, verbose=True):
    """Play every pair both ends. Returns per-entrant records and the table."""
    from .watch import resolve_policy

    policies = {}
    for s in specs:
        pol, _ = resolve_policy(s)
        policies[_name(s)] = pol
    names = list(policies)

    gf = {n: 0 for n in names}
    ga = {n: 0 for n in names}
    pairs = {}

    for a, b in itertools.combinations(names, 2):
        # Both orderings: A at the +x end, then B at the +x end.
        s1, _ = play(policies[a], policies[b], num_envs=envs, steps=steps, seed=seed)
        s2, _ = play(policies[b], policies[a], num_envs=envs, steps=steps, seed=seed + 1)
        a_goals = s1["goals_a"] + s2["goals_b"]
        b_goals = s1["goals_b"] + s2["goals_a"]
        gf[a] += a_goals; ga[a] += b_goals
        gf[b] += b_goals; ga[b] += a_goals
        pairs[f"{a} vs {b}"] = [int(a_goals), int(b_goals)]
        if verbose:
            total = a_goals + b_goals
            share = a_goals / total if total else 0.5
            se = np.sqrt(share * (1 - share) / total) if total else float("nan")
            decisive = total > 0 and abs(share - 0.5) > 1.96 * se
            verdict = ("" if not decisive else ("  <- " + (a if share > 0.5 else b) + " better"))
            print(f"  {a:28s} {a_goals:4d} - {b_goals:<4d} {b:28s}{verdict}", flush=True)

    table = []
    for n in names:
        total = gf[n] + ga[n]
        share = gf[n] / total if total else 0.5
        table.append({"name": n, "goals_for": gf[n], "goals_against": ga[n],
                      "goal_share": round(float(share), 4),
                      "goal_diff": gf[n] - ga[n]})
    table.sort(key=lambda r: -r["goal_share"])
    return table, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs="*", default=[], help="extra checkpoints to include")
    ap.add_argument("--only", nargs="*", default=None, help="use exactly these instead")
    ap.add_argument("--envs", type=int, default=96)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default=None, help="write results here")
    args = ap.parse_args()

    if args.quick:
        args.envs, args.steps = 48, 250

    if args.only:
        specs = args.only
    else:
        specs = sorted(glob.glob("checkpoints/*.pt")) + ["chase", "random"] + args.add
    if len(specs) < 2:
        raise SystemExit("need at least two entrants")

    print(f"ladder: {len(specs)} entrants, {args.envs} envs x {args.steps} steps per side\n")
    table, pairs = run_ladder(specs, args.envs, args.steps, args.seed)

    print(f"\n{'rank':<5} {'entrant':<30} {'GF':>5} {'GA':>5} {'diff':>6} {'share':>7}")
    for i, r in enumerate(table, 1):
        print(f"{i:<5} {r['name']:<30} {r['goals_for']:>5} {r['goals_against']:>5} "
              f"{r['goal_diff']:>+6} {r['goal_share']:>7.3f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"table": table, "pairs": pairs,
                       "envs": args.envs, "steps": args.steps}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
