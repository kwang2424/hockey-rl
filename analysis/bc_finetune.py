#!/usr/bin/env python3
"""Pre-registered screen: does PPO improve on a SniperBot copy without
losing it?

    PYTHONPATH=. python3 analysis/bc_finetune.py runs/bc-ft

Committed before any run of this screen existed. Do not edit after looking
at results.

Start: checkpoints/bc-sniper.pt, behaviour-cloned from SniperBot (rl/bc.py,
1.02M labelled states, seed 0). Measured before this screen, solid-net
full-skill goalie game: stochastic self-play 11 goals / 177 shots (0.062 per
shot), shots from a median 6.2 m at 7.8 deg error; never shoots with
deterministic actions; loses 0-75 to SniperBot both ends.

Design: 4 runs, seeds 30-33, each --init-from checkpoints/bc-sniper.pt,
--lr 1e-4, --total-steps 10M, solid_net=1, goalie=1, no goalie curriculum
(full skill throughout), everything else default. Anchors sniper, chase,
random, v8.

All measurements in the full-skill goalie game, on each run's final.pt:

PRIMARY 1 (keeps scoring): PASS iff >= 3 of 4 runs have stochastic self-play
goals per shot >= 0.05 (128 envs x 600 steps, seed 11).
PRIMARY 2 (improves): PASS iff >= 3 of 4 runs beat the BC start head to head,
both ends (96 envs x 400 steps per end), goal share > 0.5.

SECONDARY (no verdict): each run's goals and shots in self-play, shot
distance and heading error, and head to head against SniperBot, ChaseBot and
v8; the same for the BC start.
"""
import os, sys

from analysis.goalie_v9 import CFG, pol, selfplay
from analysis.aim_geometry import geometry
from analysis.hold_rule import h2h

START = "checkpoints/bc-sniper.pt"
RUNS = [f"s{i}" for i in range(30, 34)]


def report(label, p):
    sp = selfplay(p)
    g = geometry(p, p, CFG)
    hs = {o: h2h(p, pol(o), CFG) for o in ("sniper", "chase", "checkpoints/v8-screened-50m.pt")}
    print(f"  {label}: self-play {sp['goals']}/{sp['shots']} (goals/shot {sp['gps']:.3f}, goals/ep {sp['goals_per_ep']:.3f})"
          f"  shot dist {g['dist']:.1f} m, err {g['med']:.1f} deg  | "
          + "  ".join(f"vs {os.path.basename(o).split('-')[0]} {a}-{b}" for o, (a, b) in hs.items()), flush=True)
    return sp


def main():
    root = sys.argv[1]
    start = pol(START)
    print("BC start:")
    report("bc-sniper", start)
    keeps = improves = 0
    print("\nfine-tuned:")
    for r in RUNS:
        p = pol(os.path.join(root, r, "final.pt"))
        sp = report(r, p)
        a, b = h2h(p, start, CFG)
        share = a / max(a + b, 1)
        keeps += sp["gps"] >= 0.05
        improves += share > 0.5
        print(f"      vs BC start {a}-{b} (share {share:.3f})", flush=True)
    print(f"\nPRIMARY 1 (keeps scoring): {keeps} of {len(RUNS)}  ->  {'PASS' if keeps >= 3 else 'FAIL'}")
    print(f"PRIMARY 2 (beats BC start): {improves} of {len(RUNS)}  ->  {'PASS' if improves >= 3 else 'FAIL'}")


if __name__ == "__main__":
    main()
