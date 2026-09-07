#!/usr/bin/env python3
"""Export a game to a JSON trajectory that any engine can replay.

    python -m hockey.export --a runs/v1/best.pt --b chase --out game.json

The sim is already decoupled from rendering, so "visualise it in Godot/Unity"
does not mean porting the physics -- it means dumping state and replaying it.
That is a much better arrangement than simulating inside the engine: training
keeps its ~55k steps/sec, and the viewer can be as pretty as you like without
costing a single step of throughput.

Schema (v1), all lengths in metres, angles in radians, +y is up:

{
  "format": "hockey-rl-trajectory",
  "version": 1,
  "fps": 30.0,
  "rink": {                      # everything needed to draw the surface
    "length": 60.0, "width": 26.0, "corner_radius": 8.5,
    "goal_line_x": 26.0, "goal_half_width": 0.915, "goal_depth": 1.12,
    "skater_radius": 0.45, "puck_radius": 0.038, "blade_offset": 0.85
  },
  "teams": [{"name": "...", "colour": "#d63e3e", "attacks_x": 1}, ...],
  "frames": [
    {
      "t": 0.0333,               # seconds
      "skaters": [[x, y, theta], [x, y, theta]],   # index 0 = team 0
      "puck": [x, y],
      "possessor": -1,           # -1 loose, else skater index
      "score": [0, 0],
      "event": "goal_a"          # optional; omitted on ordinary frames
    }
  ]
}

Frames are absolute state, not deltas, so a viewer can scrub to any index
without replaying from the start.
"""

import argparse
import json

import numpy as np

from .config import DEFAULT, Config
from .env import VecHockeyEnv

SCHEMA_VERSION = 1


def record(policy_a, policy_b, steps=900, seed=0, cfg: Config = DEFAULT,
           names=("A", "B"), deterministic=True):
    """Play one game and return it as a JSON-serialisable dict."""
    env = VecHockeyEnv(num_envs=1, cfg=cfg, seed=seed)
    obs = env.observe()
    score = [0, 0]
    frames = []

    for i in range(steps):
        act = np.zeros((1, 2, 3))
        act[:, 0] = policy_a.act(obs[:, 0], deterministic=deterministic)
        act[:, 1] = policy_b.act(obs[:, 1], deterministic=deterministic)

        frame = {
            "t": round(i * cfg.control_dt, 4),
            "skaters": [[round(float(env.skater_pos[0, a, 0]), 4),
                         round(float(env.skater_pos[0, a, 1]), 4),
                         round(float(env.theta[0, a]), 4)] for a in range(2)],
            "puck": [round(float(env.puck_pos[0, 0]), 4),
                     round(float(env.puck_pos[0, 1]), 4)],
            "possessor": int(env.possessor[0]),
            "score": list(score),
        }

        obs, _, goal, trunc, info = env.step(act)
        if info["goal_a"][0]:
            score[0] += 1
            frame["event"] = "goal_a"
        elif info["goal_b"][0]:
            score[1] += 1
            frame["event"] = "goal_b"
        elif trunc[0]:
            frame["event"] = "period_end"
        frames.append(frame)

    return {
        "format": "hockey-rl-trajectory",
        "version": SCHEMA_VERSION,
        "fps": round(1.0 / cfg.control_dt, 4),
        "rink": {
            "length": cfg.rink_length, "width": cfg.rink_width,
            "corner_radius": cfg.corner_radius,
            "goal_line_x": cfg.goal_line_x,
            "goal_half_width": cfg.goal_half_width,
            "goal_depth": cfg.goal_depth,
            "skater_radius": cfg.skater_radius,
            "puck_radius": cfg.puck_radius,
            "blade_offset": cfg.blade_offset,
        },
        "teams": [
            {"name": str(names[0]), "colour": "#d63e3e", "attacks_x": 1},
            {"name": str(names[1]), "colour": "#2e6ed6", "attacks_x": -1},
        ],
        "final_score": list(score),
        "frames": frames,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="chase", help="team A policy: path.pt | chase | random | still")
    ap.add_argument("--b", default="chase", help="team B policy")
    ap.add_argument("--out", default="game.json")
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--indent", type=int, default=None,
                    help="pretty-print (larger files); default is compact")
    args = ap.parse_args()

    from .watch import resolve_policy
    pa, na = resolve_policy(args.a)
    pb, nb = resolve_policy(args.b)

    traj = record(pa, pb, steps=args.steps, seed=args.seed, names=(na, nb),
                  deterministic=not args.stochastic)
    with open(args.out, "w") as f:
        json.dump(traj, f, indent=args.indent)

    import os
    size = os.path.getsize(args.out) / 1024.0
    print(f"{na} vs {nb}: {traj['final_score'][0]}-{traj['final_score'][1]}")
    print(f"wrote {args.out}  ({len(traj['frames'])} frames, {size:.0f} KB, "
          f"{traj['fps']:.0f} fps)")


if __name__ == "__main__":
    main()
