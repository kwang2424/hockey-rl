#!/usr/bin/env python3
"""Pre-registered screen: two ways out of the stuck state in the fixed game.

    PYTHONPATH=. python3 analysis/escape_screen.py runs/escape

Committed before any run of this screen existed. Do not edit after looking
at results.

Why: in goalie-v9, 12 of 12 from-scratch networks never scored in the
solid-net goalie game and settled into the retreat-and-wait stuck state.
Without the net hole's free goals there is no goal signal to start learning.

Game: solid_net=1, goalie=1, goalie_curriculum_steps=2M (the training goalie
reaches full skill by 2M of the 3M steps). Every measurement below is in the
full-skill goalie game. 8 runs per arm, 3M steps each (--total-steps 3M).

  chances   from scratch (init_seed 500-507, seed 17), plus
            --curriculum-start 0.5 --curriculum-end 0.0 (half the resets
            start with the puck on a skater's stick, 6-20 m out facing the
            net, defender goal-side; annealed to 0 over the first 60%) and
            proximity_rate=0.003 (per-step edge for being nearer the puck;
            got every run to the puck in screen-prox)
  warm      --init-from checkpoints/v8-screened-50m.pt (widened to see the
            goalies, starting out playing exactly like v8), lr 1e-4, seeds
            20-27; no other change

Per run, on final.pt, self-play (128 envs x 600 steps, seed 11, stochastic):
  "scores"  goals per episode >= 0.05

PRIMARY, per arm: PASS iff >= 4 of 8 runs score. (goalie-v9 stage 1, plain
from scratch: 0 of 12.)

SECONDARY (reported, no verdict): per run, goals per shot, anchored rating
(sniper, chase, random, v8), shot distance and heading error; arm vs arm on
goals per shot by permutation test (run is the unit); the best run of each
arm head to head against SniperBot.
"""
import os, sys
import numpy as np

from analysis.goalie_v9 import CFG, pol, selfplay, anchored_rating
from analysis.aim_geometry import geometry
from analysis.hold_rule import h2h

ARMS = {"chances": [f"s{i}" for i in range(500, 508)],
        "warm": [f"s{i}" for i in range(20, 28)]}


def perm(a, b, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    hits = 0
    for _ in range(n):
        rng.shuffle(pool)
        hits += abs(pool[:len(a)].mean() - pool[len(a):].mean()) >= abs(obs) - 1e-12
    return obs, (hits + 1) / (n + 1)


def main():
    root = sys.argv[1]
    gps = {}
    best = {}
    for arm, runs in ARMS.items():
        print(f"\n{arm}:")
        n_score, gps[arm] = 0, []
        for r in runs:
            p = pol(os.path.join(root, arm, r, "final.pt"))
            sp = selfplay(p)
            rt, counts = anchored_rating(p)
            g = geometry(p, p, CFG)
            scores = sp["goals_per_ep"] >= 0.05
            n_score += scores
            gps[arm].append(sp["gps"])
            if arm not in best or rt > best[arm][0]:
                best[arm] = (rt, r, p)
            print(f"  {r}: goals/ep {sp['goals_per_ep']:.3f}  goals/shot {sp['gps']:.3f} ({sp['goals']}/{sp['shots']})  "
                  f"rating {rt:+.3f}  vs sniper {counts['sniper'][0]}-{counts['sniper'][1]}  "
                  f"shot dist {g['dist']:.1f} m  heading err {g['med']:.1f} deg  {'SCORES' if scores else 'stuck'}", flush=True)
        print(f"PRIMARY {arm}: {n_score} of {len(runs)} runs score  ->  {'PASS' if n_score >= 4 else 'FAIL'}")
    d, pv = perm(gps["warm"], gps["chances"])
    print(f"\nsecondary: goals/shot warm - chances = {d:+.3f}, permutation p = {pv:.3f}")
    for arm, (rt, r, p) in best.items():
        a, b = h2h(p, pol("sniper"), CFG)
        print(f"  best {arm} ({r}, rating {rt:+.3f}) vs SniperBot, both ends: {a}-{b}")


if __name__ == "__main__":
    main()
