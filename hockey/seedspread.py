#!/usr/bin/env python3
"""How different are two runs that differ only by seed?

    python -m hockey.seedspread runs/exp-sigma-ctl runs/exp-seed-1 runs/exp-seed-2

Why this exists
---------------

Every verdict in EXPERIMENTS.md -- six losses, one win, one null -- was
recorded by laddering a changed run against an unchanged one and reading the
goal share. None of them had an error bar, so none of them could distinguish
"this change did something" from "these are two runs".

That gap stopped being hypothetical: exp-sigma-ctl and exp-100m are the same
config and the same seed on different containers, and at 40M one beat v2 77-54
while the other lost 34-59. If same-config runs can differ by that much, a
verdict needs to clear that much before it means anything.

So: take N runs that differ only by seed, play their same-milestone snapshots
against each other, and report the spread of goal share. Anything inside that
band is a seed difference wearing a result's clothing.

Two sources of noise are separated because they have different remedies:

- **Eval noise** -- the ladder is a finite sample. Measured by replaying one
  frozen pair under several eval seeds; shrink it with more envs or steps.
- **Seed noise** -- the runs themselves differ. Measured here; it cannot be
  shrunk by evaluating harder, only by running more seeds.
"""

import argparse
import glob
import itertools
import os

import numpy as np

from .ladder import run_ladder


def milestones(run_dirs):
    """Archive milestones present in *every* run, so each comparison is like
    for like. A run killed early simply contributes fewer rows."""
    sets = []
    for d in run_dirs:
        found = {os.path.basename(p) for p in glob.glob(os.path.join(d, "archive", "*.pt"))}
        if os.path.exists(os.path.join(d, "final.pt")):
            found.add("final.pt")
        sets.append(found)
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


def spread_at(run_dirs, milestone, envs, steps, seed):
    """Every pairing of the same milestone across runs. Returns goal shares."""
    specs = []
    for d in run_dirs:
        p = (os.path.join(d, "final.pt") if milestone == "final.pt"
             else os.path.join(d, "archive", milestone))
        specs.append(p)
    _, pairs = run_ladder(specs, envs=envs, steps=steps, seed=seed, verbose=False)
    return [a / (a + b) for a, b in pairs.values() if a + b > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories differing only by seed")
    ap.add_argument("--envs", type=int, default=96)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if len(args.runs) < 2:
        raise SystemExit("need at least two runs to have a spread")

    ms = milestones(args.runs)
    if not ms:
        raise SystemExit("no milestone is present in all of these runs")

    n_pairs = len(list(itertools.combinations(args.runs, 2)))
    print(f"{len(args.runs)} runs, {n_pairs} pairings per milestone, "
          f"{args.envs} envs x {args.steps} steps per side\n")
    print(f"{'milestone':<18} {'n':>2} {'mean':>6} {'sd':>6} {'min':>6} {'max':>6} "
          f"{'widest gap':>11}")

    everything = []
    for m in ms:
        shares = spread_at(args.runs, m, args.envs, args.steps, args.seed)
        if not shares:
            continue
        everything += shares
        s = np.array(shares)
        # Each pairing is one ordered result; its mirror is 1-s by construction,
        # so the spread that matters is how far from an even split it can land.
        gap = max(s.max(), 1 - s.min())
        print(f"{m:<18} {len(s):>2} {s.mean():>6.3f} {s.std(ddof=1):>6.3f} "
              f"{s.min():>6.3f} {s.max():>6.3f} {gap:>11.3f}")

    a = np.array(everything)
    lo, hi = np.percentile(a, [5, 95])
    print(f"\nAcross all milestones: {len(a)} same-config pairings")
    print(f"  goal share      mean {a.mean():.3f}  sd {a.std(ddof=1):.3f}")
    print(f"  5-95 percentile [{lo:.3f}, {hi:.3f}]")
    print(f"  widest observed [{a.min():.3f}, {a.max():.3f}]")
    print(f"\nA one-change verdict must clear this band to mean anything.")


if __name__ == "__main__":
    main()
