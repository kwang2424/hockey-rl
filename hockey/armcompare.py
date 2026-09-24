#!/usr/bin/env python3
"""Compare two arms of an experiment when the unit of analysis is the run.

    python -m hockey.armcompare --a runs/exp-seed-* --b runs/exp-curric-*

Why this exists
---------------

exp-seeds measured what two runs of the *same* config do head to head: goal
shares from 0.20 to 0.93, and one run in four collapsing outright. Every
verdict recorded in EXPERIMENTS.md before that was one run against one run,
which cannot separate a change from a seed.

The fix is not a better metric, it is more runs and a test that respects them.
Two things follow:

- **The run is the unit, not the pairing.** Six runs against six runs gives 36
  pairings, but not 36 independent observations: all six pairings involving one
  run share that run's luck. Treating pairings as independent would shrink the
  error bar by a factor it has not earned.
- **The null is available empirically.** Ladder every run against every other
  one, then shuffle the arm labels. Each shuffle gives a mean cross-label share
  under "the label means nothing", which is exactly the null hypothesis, drawn
  from this experiment's own runs rather than assumed. No normality, no
  variance estimate, and run-level correlation is preserved because whole runs
  move between labels together.

A collapsed run is reported, never dropped. It is a real outcome of the config
under test -- if one arm collapses more often, that is the finding, and it
would vanish under any rule that excluded it for being extreme.
"""

import argparse
import os

import numpy as np

from .ladder import run_ladder, unique_names


def share_matrix(specs, names, envs, steps, seed):
    """Full round-robin. Returns S where S[i][j] is i's goal share against j."""
    table, pairs = run_ladder(specs, envs=envs, steps=steps, seed=seed, verbose=False)
    n = len(names)
    S = np.full((n, n), np.nan)
    idx = {nm: i for i, nm in enumerate(names)}
    for key, (ga, gb) in pairs.items():
        a, b = key.split(" vs ")
        if ga + gb == 0:
            continue
        i, j = idx[a], idx[b]
        S[i][j] = ga / (ga + gb)
        S[j][i] = gb / (ga + gb)
    return S, table


def cross_mean(S, label):
    """Mean share of label-0 runs against label-1 runs, over defined pairs."""
    a = np.where(label == 0)[0]
    b = np.where(label == 1)[0]
    vals = S[np.ix_(a, b)]
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def permutation_p(S, label, perms, rng):
    """Two-sided p: how often does relabelling produce an effect this large?"""
    observed = cross_mean(S, label)
    n_a = int((label == 0).sum())
    null = np.empty(perms)
    lab = np.zeros(len(label), dtype=int)
    for k in range(perms):
        pick = rng.permutation(len(label))[:n_a]
        lab[:] = 1
        lab[pick] = 0
        null[k] = cross_mean(S, lab)
    # Centre on 0.5 rather than on the null mean: 0.5 IS no difference here,
    # and the null is symmetric about it by construction.
    p = float((np.abs(null - 0.5) >= abs(observed - 0.5) - 1e-12).mean())
    return observed, p, null


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True, help="arm A run directories")
    ap.add_argument("--b", nargs="+", required=True, help="arm B run directories")
    ap.add_argument("--milestone", default="final.pt",
                    help="final.pt, or an archive basename like step_040000000.pt")
    ap.add_argument("--envs", type=int, default=96)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perms", type=int, default=20000)
    ap.add_argument("--collapse-below", type=float, default=0.15,
                    help="overall goal share under which a run is called collapsed")
    args = ap.parse_args()

    def spec(d):
        return (os.path.join(d, "final.pt") if args.milestone == "final.pt"
                else os.path.join(d, "archive", args.milestone))

    specs = [spec(d) for d in args.a] + [spec(d) for d in args.b]
    missing = [s for s in specs if not os.path.exists(s)]
    if missing:
        raise SystemExit("missing checkpoints:\n  " + "\n  ".join(missing))

    names = unique_names(specs)
    label = np.array([0] * len(args.a) + [1] * len(args.b))

    print(f"arm A: {len(args.a)} runs   arm B: {len(args.b)} runs   "
          f"milestone {args.milestone}")
    print(f"{len(specs)} entrants, {len(specs)*(len(specs)-1)//2} pairings, "
          f"{args.envs} envs x {args.steps} steps per side\n")

    S, table = share_matrix(specs, names, args.envs, args.steps, args.seed)

    by_name = {r["name"]: r for r in table}
    print(f"{'run':<34} {'arm':>3} {'share vs field':>15}")
    collapsed = {0: 0, 1: 0}
    for nm, lb in zip(names, label):
        sh = by_name[nm]["goal_share"]
        flag = ""
        if sh < args.collapse_below:
            collapsed[lb] += 1
            flag = "  <- collapsed"
        print(f"{nm:<34} {'AB'[lb]:>3} {sh:>15.3f}{flag}")
    print(f"\ncollapsed runs:  A {collapsed[0]}/{len(args.a)}   "
          f"B {collapsed[1]}/{len(args.b)}")

    rng = np.random.default_rng(args.seed)
    observed, p, null = permutation_p(S, label, args.perms, rng)
    lo, hi = np.percentile(null, [2.5, 97.5])

    print(f"\nA vs B mean goal share (A's perspective): {observed:.3f}")
    print(f"  0.500 is no difference")
    print(f"  permutation null: 95% of relabellings fall in [{lo:.3f}, {hi:.3f}]")
    print(f"  two-sided p = {p:.4f}  ({args.perms:,} permutations)")
    verdict = ("DIFFERENCE" if p < 0.05 else
               "no detectable difference at this sample size")
    print(f"\n  verdict: {verdict}")
    if p >= 0.05:
        half = (hi - lo) / 2
        print(f"  an effect smaller than about {half:.2f} in goal share cannot be\n"
              f"  resolved with {len(args.a)}v{len(args.b)} runs; that is a limit of n, "
              f"not evidence of equality.")


if __name__ == "__main__":
    main()
