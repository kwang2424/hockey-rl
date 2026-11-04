#!/usr/bin/env python3
"""POST-HOC, written after analysis/hold_aim.py's result (FAIL: 1.5% on target).
It cannot change that verdict. It asks why aiming has not moved in four
experiments, by measuring shot geometry directly.

"On target" means the shot's straight line crosses a 1.83 m goal mouth --
about +/-2-3.5 degrees of heading from 15-25 m. puck_goal_speed_rate pays the
puck's velocity toward the goal, so a shot 20 degrees off still earns 94% of
the maximum. If the reward taught rough aim, on-target would not see it.

Per policy, self-play, team A, stochastic actions, seeded:
  heading error at release  angle between facing and the line to the goal
  shot distance             from the shooter to the goal centre
  possessions               ended by a shot, or lost before one

    PYTHONPATH=. python3 analysis/aim_geometry.py <experiment_dir>
"""
import os, sys
import numpy as np, torch

from hockey.config import DEFAULT, Config
from hockey.env import VecHockeyEnv
from hockey.watch import resolve_policy


def geometry(pol_a, pol_b, cfg, envs=128, steps=900, seed=11):
    torch.manual_seed(0); np.random.seed(0)
    env = VecHockeyEnv(num_envs=envs, cfg=cfg, seed=seed); obs = env.observe()
    goal = env._goal_centers[0]
    err, dist = [], []
    shot_end = lost_end = 0
    for _ in range(steps):
        act = np.stack([pol_a.act(obs[:, 0], deterministic=False),
                        pol_b.act(obs[:, 1], deterministic=False)], 1)
        held = env.possessor == 0
        pos, th = env.skater_pos[:, 0].copy(), env.theta[:, 0].copy()
        obs, *_ = env.step(act)
        f = env.fired[:, 0]
        if f.any():
            to_g = goal - pos[f]
            bearing = np.arctan2(to_g[:, 1], to_g[:, 0])
            e = np.abs((th[f] - bearing + np.pi) % (2 * np.pi) - np.pi)
            err += np.degrees(e).tolist(); dist += np.linalg.norm(to_g, axis=1).tolist()
        shot_end += int(f.sum())
        lost_end += int((held & ~f & (env.possessor != 0)).sum())
    err, dist = np.array(err), np.array(dist)
    q = lambda a, p: float(np.percentile(a, p)) if a.size else float("nan")
    return dict(n=len(err), med=q(err, 50), within10=float((err <= 10).mean()) if err.size else float("nan"),
                within30=float((err <= 30).mean()) if err.size else float("nan"),
                dist=q(dist, 50), shot_share=shot_end / max(shot_end + lost_end, 1))


def main():
    root = sys.argv[1]
    rule = Config(min_hold_time=0.5)
    chase, _ = resolve_policy("chase")
    pols = [("v8, ordinary game", "checkpoints/v8-screened-50m.pt", DEFAULT),
            ("hold05 start point", "checkpoints/hold05-v8-finetune.pt", rule),
            ("rule (20M more)", os.path.join(root, "rule", "final.pt"), rule),
            ("rule_aim (20M more)", os.path.join(root, "rule_aim", "final.pt"), rule)]
    print(f"{'self-play':<22} {'shots':>6} {'median err':>10} {'<=10 deg':>9} {'<=30 deg':>9} {'median dist':>11} {'poss ending in a shot':>22}")
    rows = [(lab, resolve_policy(s)[0], c) for lab, s, c in pols] + [("ChaseBot, ordinary game", chase, DEFAULT)]
    for lab, p, c in rows:
        g = geometry(p, p, c)
        print(f"{lab:<22} {g['n']:>6} {g['med']:>9.1f}° {g['within10']:>9.0%} {g['within30']:>9.0%} "
              f"{g['dist']:>10.1f}m {g['shot_share']:>22.0%}", flush=True)
    print("(uniformly random aim: median error 90 deg, 6% within 10, 17% within 30)")


if __name__ == "__main__":
    main()
