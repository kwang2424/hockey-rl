#!/usr/bin/env python3
"""Pre-registered analysis: does an aiming reward work once holding is forced?

    PYTHONPATH=. python3 analysis/hold_aim.py <experiment_dir>

Committed before any run of this screen existed. Do not edit after looking at
results.

Why: under the settle-before-shooting rule (min_hold_time=0.5) the agent now
holds the puck, but its shots stay blind (0.8% on target) -- the only signal
for which way to shoot is a goal many steps later. An aiming reward
(puck_goal_speed_rate) failed before because the agent never held the puck
long enough to collect it; under the rule it does. The skater can see where
the net is (atk_goal_rel is in its observation), so aiming is learnable.

Design: two arms, each fine-tuned from checkpoints/hold05-v8-finetune.pt for
20M steps (lr 1e-4, seed 17), both under min_hold_time=0.5, one setting apart:
  rule      no change
  rule_aim  puck_goal_speed_rate = 0.01
One run per arm: a screen.

Everything is measured under the 0.5 s rule (the game both arms train in).

PRIMARY: self-play shots on target, rule_aim against rule. Pass iff rule_aim
reaches >= 15% on target AND beats rule by >= 10 points.

SECONDARY: possession time, shots and goals per minute in self-play; head to
head rule_aim vs rule; each arm vs ChaseBot, and ChaseBot's possession and
shots against each.
"""
import os, sys
import numpy as np, torch

from hockey.config import Config
from hockey.diagnose import diagnose
from hockey.watch import resolve_policy
from analysis.hold_rule import beh, h2h, line

CFG = Config(min_hold_time=0.5)


def main():
    root = sys.argv[1]
    pol = {a: resolve_policy(os.path.join(root, a, "final.pt"))[0] for a in ("rule", "rule_aim")}
    chase, _ = resolve_policy("chase")
    b = {a: beh(pol[a], CFG) for a in pol}
    print("self-play under the 0.5 s rule:")
    for a in pol:
        print(line(a, b[a]))
    t, c = b["rule_aim"]["shot_on_target_frac"], b["rule"]["shot_on_target_frac"]
    ok = t >= 0.15 and (t - c) >= 0.10
    print(f"\nPRIMARY: on target rule_aim {t:.1%} vs rule {c:.1%} (gap {t - c:+.1%})  ->  {'PASS' if ok else 'FAIL'}")
    x, y = h2h(pol["rule_aim"], pol["rule"], CFG)
    print(f"\nsecondary: rule_aim vs rule {x}-{y} (share {x / max(x + y, 1):.3f})")
    for a in pol:
        g, h = h2h(pol[a], chase, CFG)
        torch.manual_seed(0); np.random.seed(0)
        cb = diagnose(chase, pol[a], num_envs=128, steps=600, seed=11, cfg=CFG)
        print(f"  {a:<9} vs ChaseBot {g}-{h} (share {g / max(g + h, 1):.3f});  "
              f"ChaseBot possession {cb['possession_frac']:.1%}, shots {cb['shots']}")


if __name__ == "__main__":
    main()
