#!/usr/bin/env python3
"""POST-HOC, written after seeing analysis/hold_rule.py's results. Not
pre-registered; it cannot change that screen's verdict (both arms FAILED the
learned-aiming test).

Two questions the results raised:
  1. The 0.5 s arm scores the most goals while hitting the net least. Are its
     goals carried in -- the scorer still holding the puck the step before it
     crosses the line -- rather than shot?
  2. How much does each rule handicap ChaseBot by itself? Calibrates the
     head-to-head numbers against it.

    PYTHONPATH=. python3 analysis/hold_rule_posthoc.py <experiment_dir>
"""
import os, sys
import numpy as np, torch

from hockey.config import DEFAULT, Config
from hockey.env import VecHockeyEnv
from hockey.watch import resolve_policy
from analysis.hold_rule import h2h


def goal_types(pol, cfg, envs=128, steps=1200, seed=11):
    """Self-play; classify each goal by whether the scorer held the puck on the
    step it went in (carried) or it was loose (shot, deflection, bounce)."""
    torch.manual_seed(0); np.random.seed(0)
    env = VecHockeyEnv(num_envs=envs, cfg=cfg, seed=seed); obs = env.observe()
    carried = loose = 0
    for _ in range(steps):
        act = np.stack([pol.act(obs[:, a], deterministic=False) for a in (0, 1)], 1)
        before = env.possessor.copy()
        obs, _, goal, _, info = env.step(act)
        for team, key in ((0, "goal_a"), (1, "goal_b")):
            g = info[key]
            carried += int((g & (before == team)).sum())
            loose += int((g & (before != team)).sum())
    return carried, loose


def main():
    root = sys.argv[1]
    arms = {"ctl": None, "hold02": 0.2, "hold05": 0.5}
    pol = {a: resolve_policy(os.path.join(root, a, "final.pt"))[0] for a in arms}
    v8, _ = resolve_policy("checkpoints/v8-screened-50m.pt")
    chase, _ = resolve_policy("chase")

    print("1. how goals are scored (self-play):")
    for arm, t in [("ctl", 0.0), ("hold02", 0.2), ("hold05", 0.5), ("ctl", 0.5)]:
        c, l = goal_types(pol[arm], Config(min_hold_time=t))
        print(f"  {arm:<7} rule {t:.1f} s   carried in {c:>4}   loose {l:>4}   carried share {c / max(c + l, 1):.0%}")
    c, l = goal_types(chase, DEFAULT)
    print(f"  ChaseBot, ordinary game   carried in {c:>4}   loose {l:>4}   carried share {c / max(c + l, 1):.0%}")

    print("\n2. ChaseBot under each rule, against the same opponents:")
    for t in (0.0, 0.2, 0.5):
        cfg = Config(min_hold_time=t)
        for name, p in (("v8", v8), ("ctl", pol["ctl"])):
            a, b = h2h(p, chase, cfg)
            print(f"  rule {t:.1f} s   {name:<4} vs ChaseBot {a:>4}-{b:<4} share {a / max(a + b, 1):.3f}")
        if t:
            arm = "hold02" if t == 0.2 else "hold05"
            a, b = h2h(pol[arm], chase, cfg)
            print(f"  rule {t:.1f} s   {arm} vs ChaseBot {a:>4}-{b:<4} share {a / max(a + b, 1):.3f}")


if __name__ == "__main__":
    main()
