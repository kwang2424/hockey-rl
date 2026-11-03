#!/usr/bin/env python3
"""Pre-registered: does training in the fixed game (solid net + goalie) produce
an agent that carries in and aims?

    PYTHONPATH=. python3 analysis/goalie_v9.py select runs/goalie-v9
    PYTHONPATH=. python3 analysis/goalie_v9.py final  runs/goalie-v9

Committed before any run of this experiment existed. Do not edit after
looking at results.

Why: 90% of the learned agents' goals entered the net from behind or the
side (EXPERIMENTS.md, net-hole), so aiming never paid, and no reward or rule
produced it. With a solid net and a scripted goalie, only close, aimed shots
score (analysis/goalie_calibration.py). The question is whether ordinary
self-play PPO now learns that.

Design -- the v8 screening pipeline, in the new game, from scratch:
  game      solid_net=1, goalie=1; everything else at its default (no
            settle-before-shooting rule: whether it is still needed once
            aiming pays is a separate question)
  goalie    curriculum goalie_curriculum_steps=15M: the training goalies'
            skill rises linearly from 0 (static) to full over the first 15M
            steps. Every evaluation faces the full-skill goalie.
  anchors   sniper, chase, random, v8 -- the rating is re-anchored on
            SniperBot, because the old anchors were measured in the broken game
  networks  12 starting networks, init_seed 400-411, seed 17, each run as the
            literal start of a 50M-step run (--total-steps 50M --stop-at 3M)
  stage 2   at 3M, keep the one network with the highest anchored rating (vs
            sniper, chase, random, v8, full goalie game, 128 envs x 600 steps,
            one end, seed 12345); ties broken by self-play goals per shot.
            No stage-1 critic filter: its cutoff was fitted in the old game,
            and 12 x 3M is affordable here.
  stage 3   resume the selected network to 50M (final.pt = "v9").

Everything below is measured in the full-skill goalie game with a solid net.

PRIMARY: v9 (final.pt) self-play goals per shot. PASS iff >= 0.05.
  (v8 0.006, ChaseBot 0.024, SniperBot 0.58 in the same game; 0.05 is about
  8x v8 and 2x ChaseBot, and a rough or long shot almost never scores.)

SECONDARY (reported, no verdict):
  * head to head, both ends, v9 against SniperBot, ChaseBot, v8, random, and
    against its own rating-selected best.pt
  * self-play behaviour: shots on target, mean possession, shot distance,
    heading error at release, share of possessions ending in a shot
  * stage 2: how many of the 12 networks score at all at 3M (self-play goals
    per episode >= 0.05), the stuck rate in the new game

One selected run: this demonstrates what the pipeline can produce, it is not
a comparison of training settings.
"""
import os, sys

import numpy as np
import torch
from dataclasses import replace

from hockey.config import DEFAULT
from hockey.rating import rating
from hockey.rollout import play
from hockey.watch import resolve_policy
from analysis.aim_geometry import geometry
from analysis.hold_rule import beh, h2h

CFG = replace(DEFAULT, solid_net=True, goalie=True)
NETS = list(range(400, 412))
ANCHORS = ("sniper", "chase", "random", "checkpoints/v8-screened-50m.pt")
PASS_GOALS_PER_SHOT = 0.05


def pol(spec):
    return resolve_policy(spec)[0]


def anchored_rating(p):
    counts = {}
    for spec in ANCHORS:
        s, _ = play(p, pol(spec), num_envs=128, steps=600, seed=12345, cfg=CFG)
        counts[spec] = (s["goals_a"], s["goals_b"])
    return rating(counts), counts


def selfplay(p):
    torch.manual_seed(0); np.random.seed(0)
    s, _ = play(p, p, num_envs=128, steps=600, seed=11, cfg=CFG, deterministic=False)
    goals, shots = s["goals_a"] + s["goals_b"], s["shots_a"] + s["shots_b"]
    return dict(goals=goals, shots=shots, gps=goals / max(shots, 1),
                goals_per_ep=goals / max(s["episodes"], 1))


def select(root):
    rows = []
    for i in NETS:
        path = os.path.join(root, f"net{i}", "latest.pt")
        p = pol(path)
        r, counts = anchored_rating(p)
        sp = selfplay(p)
        rows.append((r, sp["gps"], i, counts, sp))
        print(f"  net{i}: rating {r:+.3f}  "
              + "  ".join(f"{os.path.basename(k).split('-')[0]} {a}-{b}" for k, (a, b) in counts.items())
              + f"  | self-play goals/ep {sp['goals_per_ep']:.3f}, goals/shot {sp['gps']:.3f}", flush=True)
    scoring = sum(1 for row in rows if row[4]["goals_per_ep"] >= 0.05)
    print(f"\nstage 2: {scoring} of {len(rows)} networks score at 3M (self-play goals/ep >= 0.05)")
    best = max(rows, key=lambda row: (row[0], row[1]))
    print(f"SELECTED: net{best[2]} (rating {best[0]:+.3f})")
    with open(os.path.join(root, "selected.txt"), "w") as f:
        f.write(f"{best[2]}\n")


def final(root):
    sel = int(open(os.path.join(root, "selected.txt")).read().strip())
    run = os.path.join(root, f"net{sel}")
    v9 = pol(os.path.join(run, "final.pt"))
    sp = selfplay(v9)
    ok = sp["gps"] >= PASS_GOALS_PER_SHOT
    print(f"v9 = net{sel} @ final.pt")
    print(f"PRIMARY: self-play goals per shot {sp['gps']:.3f} ({sp['goals']} goals / {sp['shots']} shots), "
          f"threshold {PASS_GOALS_PER_SHOT}  ->  {'PASS' if ok else 'FAIL'}")
    print("\nsecondary, head to head (both ends), full goalie game:")
    opps = list(ANCHORS) + [os.path.join(run, "best.pt")]
    for spec in opps:
        if not os.path.exists(spec) and spec not in ("sniper", "chase", "random"):
            continue
        a, b = h2h(v9, pol(spec), CFG)
        print(f"  v9 vs {spec:<36} {a}-{b} (share {a / max(a + b, 1):.3f})", flush=True)
    b = beh(v9, CFG)
    g = geometry(v9, v9, CFG)
    print(f"\nself-play behaviour: on target {b['shot_on_target_frac']:.1%}, possession "
          f"{b['mean_possession_sec']:.3f} s, shot distance {g['dist']:.1f} m, heading error "
          f"{g['med']:.1f} deg, possessions ending in a shot {g['shot_share']:.0%}")


if __name__ == "__main__":
    {"select": select, "final": final}[sys.argv[1]](sys.argv[2])
