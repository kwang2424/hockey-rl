#!/usr/bin/env python3
"""Pre-registered analysis for the puck-control screen.

    PYTHONPATH=. python3 analysis/puck_screen.py <experiment_dir>

Committed before any run of this screen existed. Do not edit after looking at
results.

Why: v8 fires on 99% of the steps it holds the puck, possessions last one
control step (0.03 s), and 2% of its shots are on target. Its shoot output
while holding averages +3.66 against a firing threshold of 0 -- a learned
preference, not exploration noise. Every shaping term is a potential, which
the critic cancels, so nothing rewards holding or aiming; only goals count.

Design: three arms, each fine-tuned from checkpoints/v8-screened-50m.pt for
20M steps (lr 1e-4, seed 17), identical except for one setting:
  ctl   no change
  poss  possession_rate=0.0008     per-step reward for holding the puck
  aim   puck_goal_speed_rate=0.01  per-step reward for puck velocity toward
                                   the opponent's net (negative away from it)
One run per arm: a screen. The behaviour changes sought are large; a winner
gets a replication before it counts.

Measured on each arm's final network:
  behaviour  hockey.diagnose, self-play, stochastic actions, seeded
  skill      ladder over ctl, poss, aim, v8 and ChaseBot, both ends

Per treatment arm, with the dashboard's own warning lines as the bars:
  HOLDS        mean possession >= 0.25 s
  AIMS         shots on target >= 15%
  NOT HOARDING shots >= a quarter of the control arm's
  SKILL        head-to-head goal share vs the control arm >= 0.44 (evaluation
               noise is about +/-0.06 at 2 sd)
Verdict: "fixes puck control" iff HOLDS and AIMS and NOT HOARDING; "and keeps
its skill" iff SKILL as well.
"""
import os, sys
import numpy as np, torch

from hockey.diagnose import diagnose
from hockey.ladder import run_ladder, unique_names
from hockey.watch import resolve_policy

ARMS = ("ctl", "poss", "aim")


def main():
    root = sys.argv[1]
    spec = {a: os.path.join(root, a, "final.pt") for a in ARMS}
    beh = {}
    print(f"{'arm':<5} {'possession s':>12} {'fire-while-holding':>19} {'on target':>10} {'shots':>6} {'goals/min':>10}")
    for a in ARMS:
        torch.manual_seed(0); np.random.seed(0)
        pol, _ = resolve_policy(spec[a])
        b = beh[a] = diagnose(pol, pol, num_envs=128, steps=600, seed=11)
        print(f"{a:<5} {b['mean_possession_sec']:>12.3f} {b['fire_while_holding']:>19.3f} "
              f"{b['shot_on_target_frac']:>10.1%} {b['shots']:>6} {b['goals_for_per_min']:>10.2f}", flush=True)

    specs = [spec[a] for a in ARMS] + ["checkpoints/v8-screened-50m.pt", "chase"]
    names = unique_names(specs)
    table, pairs = run_ladder(specs, envs=96, steps=400, seed=0, verbose=False)
    def share(x, y):
        k = f"{x} vs {y}"
        if k in pairs: a, b = pairs[k]
        else: b, a = pairs[f"{y} vs {x}"]
        return a / max(a + b, 1), f"{a}-{b}"
    nm = dict(zip(list(ARMS) + ["v8", "chase"], names))
    print("\nladder:")
    for r in table:
        print(f"  {r['name']:<34} share {r['goal_share']:.3f}")
    print("\nhead to head:")
    for a in ARMS:
        line = [f"vs {o}: {share(nm[a], nm[o])[1]}" for o in ARMS + ("v8", "chase") if o != a]
        print(f"  {a:<5} " + "   ".join(line))

    print("\nverdicts:")
    for a in ("poss", "aim"):
        b = beh[a]
        holds = b["mean_possession_sec"] >= 0.25
        aims = b["shot_on_target_frac"] >= 0.15
        not_hoard = b["shots"] >= 0.25 * beh["ctl"]["shots"]
        s, rec = share(nm[a], nm["ctl"])
        skill = s >= 0.44
        fixes = holds and aims and not_hoard
        v = ("fixes puck control" + (" and keeps its skill" if skill else ", but loses skill")) if fixes else "does not fix puck control"
        print(f"  {a:<5} HOLDS {holds}  AIMS {aims}  NOT-HOARDING {not_hoard}  "
              f"SKILL {skill} ({rec} vs ctl, share {s:.3f})  ->  {v}")


if __name__ == "__main__":
    main()
