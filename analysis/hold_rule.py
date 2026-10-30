#!/usr/bin/env python3
"""Pre-registered analysis for the settle-before-shooting screen.

    PYTHONPATH=. python3 analysis/hold_rule.py <experiment_dir>

Committed before any run of this screen existed. Do not edit after looking at
results.

Why: every learned agent fires on the first step it touches the puck, so it
never experiences holding, and rewards for holding or aiming are almost never
collected (puck-finetune). A minimum hold before shooting (Config.min_hold_time)
makes that experience unavoidable.

Design: three arms fine-tuned from checkpoints/v8-screened-50m.pt for 20M
steps (lr 1e-4, seed 17), one setting apart --
  ctl     no rule
  hold02  min_hold_time = 0.2 s
  hold05  min_hold_time = 0.5 s
One run per arm: a screen. A positive result is replicated before it counts.

The rule alone gives any policy extra time on the puck -- an unchanged v8 is
also forced to hold -- so LEARNED aiming is measured against the control
network playing the SAME rule:

  PRIMARY, per rule arm r:  shots on target of arm r under rule r, against
  shots on target of ctl under rule r. Pass iff arm r reaches >= 15% on target
  (the dashboard's own bar) AND beats ctl-under-r by >= 10 points.

Behaviour is hockey.diagnose, self-play, stochastic actions, seeded, 128 envs
x 600 steps, counting shots the simulator actually took.

SECONDARY: possession time and goals per minute; head to head arm r vs ctl,
both under rule r, both ends; arm r vs ChaseBot under rule r; and ctl in the
ordinary game as the v8-like baseline.
"""
import os, sys
import numpy as np, torch

from hockey.config import DEFAULT, Config
from hockey.diagnose import diagnose
from hockey.rollout import play
from hockey.watch import resolve_policy

RULES = {"hold02": 0.2, "hold05": 0.5}


def beh(pol, cfg):
    torch.manual_seed(0); np.random.seed(0)
    return diagnose(pol, pol, num_envs=128, steps=600, seed=11, cfg=cfg)


def h2h(pa, pb, cfg, envs=96, steps=400, seed=0):
    s1, _ = play(pa, pb, num_envs=envs, steps=steps, seed=seed, cfg=cfg)
    s2, _ = play(pb, pa, num_envs=envs, steps=steps, seed=seed + 1, cfg=cfg)
    a, b = s1["goals_a"] + s2["goals_b"], s1["goals_b"] + s2["goals_a"]
    return a, b


def line(label, b):
    return (f"  {label:<26} on target {b['shot_on_target_frac']:>6.1%}   possession "
            f"{b['mean_possession_sec']:.3f} s   shots {b['shots']:>4}   goals/min {b['goals_for_per_min']:.2f}")


def main():
    root = sys.argv[1]
    pol = {a: resolve_policy(os.path.join(root, a, "final.pt"))[0] for a in ("ctl", *RULES)}
    chase, _ = resolve_policy("chase")
    print("behaviour (self-play):")
    print(line("ctl, ordinary game", beh(pol["ctl"], DEFAULT)))
    for arm, t in RULES.items():
        cfg = Config(min_hold_time=t)
        bc, ba = beh(pol["ctl"], cfg), beh(pol[arm], cfg)
        print(f"\nrule {t} s:")
        print(line(f"ctl under {t} s rule", bc))
        print(line(f"{arm} under {t} s rule", ba))
        gap = ba["shot_on_target_frac"] - bc["shot_on_target_frac"]
        ok = ba["shot_on_target_frac"] >= 0.15 and gap >= 0.10
        print(f"  PRIMARY {arm}: on target {ba['shot_on_target_frac']:.1%} vs ctl {bc['shot_on_target_frac']:.1%} "
              f"(gap {gap:+.1%})  ->  {'PASS' if ok else 'FAIL'}")
        a, b = h2h(pol[arm], pol["ctl"], cfg)
        c, d = h2h(pol[arm], chase, cfg)
        print(f"  secondary: {arm} vs ctl {a}-{b} (share {a / max(a + b, 1):.3f});  {arm} vs ChaseBot {c}-{d}")


if __name__ == "__main__":
    main()
