#!/usr/bin/env python3
"""Goalie calibration: which shots does the scripted goalie stop?

    python -m analysis.goalie_calibration [--set goalie_reaction=0.15 ...]

Fires controlled shots at the +x net with nobody else in the play: the puck
sits at a spot long enough for the goalie to set, then is released at
`speed` toward a target point on the goal line. Reports the goal rate by
shot distance and target (centre of the net, or a corner just inside the
post), for aimed shots and for "rough" ones with a heading error drawn from
a normal distribution -- the learned agents' median error is ~32 degrees.

What a reasonable goalie should do (the sanity check before any training):
  * centre shots: saved from everywhere -- the goalie stands on that line;
  * aimed corner shots: mostly in from close range, mostly saved from far,
    because a far shot gives the goalie time to slide across;
  * rough shots: rarely in from anywhere.
"""

import argparse
from dataclasses import replace

import numpy as np

from hockey.config import DEFAULT
from hockey.env import VecHockeyEnv

DISTANCES = (4.0, 6.0, 8.0, 10.0, 14.0, 20.0, 26.0)
TARGETS = {"centre": 0.0, "corner": 0.78}     # metres off centre on the goal line


def shoot(cfg, dist, target_y, n, heading_sd_deg=0.0, speed=25.0, seed=0,
          angle_sd_deg=25.0, settle=0.5, horizon=90):
    """Goal rate of `n` shots from `dist` metres at a goal-line target."""
    rng = np.random.default_rng(seed)
    env = VecHockeyEnv(num_envs=n, cfg=cfg, seed=seed)
    gx = cfg.goal_line_x
    # Shot angle off the centre line, capped so the spot stays well inside
    # the boards (at 26 m only small angles are on the ice at all).
    cap = np.degrees(np.arcsin(min(1.0, (cfg.half_width - 2.0) / dist)))
    phi = np.radians(np.clip(rng.normal(0.0, angle_sd_deg, n), -min(cap, 60), min(cap, 60)))
    start = np.stack([gx - dist * np.cos(phi), dist * np.sin(phi)], axis=-1)
    side = np.where(rng.random(n) < 0.5, -1.0, 1.0)     # which corner
    aim = np.stack([np.full(n, gx), side * target_y], axis=-1) - start
    ang = np.arctan2(aim[:, 1], aim[:, 0]) + np.radians(rng.normal(0.0, heading_sd_deg, n))
    vel = speed * np.stack([np.cos(ang), np.sin(ang)], axis=-1)

    # Skaters parked at the far end, out of the play.
    env.skater_pos[:] = [[-20.0, 6.0], [-20.0, -6.0]]
    env.skater_vel[:] = 0.0
    env.omega[:] = 0.0
    env.possessor[:] = -1
    env.cooldown[:] = 1e9                     # nobody picks the puck up
    env.puck_pos[:] = start
    env.puck_vel[:] = 0.0
    env.step_count[:] = 0
    if cfg.goalie:
        env._reset_goalies(np.arange(n))
    idle = np.zeros((n, 2, 3))
    for _ in range(int(round(settle / cfg.control_dt))):   # goalie sets
        env.puck_pos[:] = start
        env.puck_vel[:] = 0.0
        env.step(idle)
        env.cooldown[:] = 1e9
    env.puck_vel[:] = vel

    scored = np.zeros(n, dtype=bool)
    live = np.ones(n, dtype=bool)
    for _ in range(horizon):
        _, _, _, _, info = env.step(idle)
        scored |= live & info["goal_a"]
        live &= ~info["episode_end"]
        env.cooldown[:] = 1e9
    return scored.mean()


def table(cfg, n=400, heading_sd_deg=0.0, speed=25.0):
    rows = {}
    for name, ty in TARGETS.items():
        rows[name] = [shoot(cfg, d, ty, n, heading_sd_deg, speed, seed=i)
                      for i, d in enumerate(DISTANCES)]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], help="Config KEY=VALUE overrides")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--speed", type=float, default=25.0)
    args = ap.parse_args()
    kw = {}
    for pair in args.set:
        k, v = pair.split("=")
        kw[k] = (v.lower() in ("1", "true", "on")) if v.lower() in ("1", "0", "true", "false", "on", "off") else float(v)
    base = replace(DEFAULT, solid_net=True, **{k: v for k, v in kw.items() if k != "goalie"})
    print("distance (m):        " + "".join(f"{d:7.0f}" for d in DISTANCES))
    for label, cfg, sd in (("no goalie, aimed", replace(base, goalie=False), 0.0),
                           ("goalie,    aimed", replace(base, goalie=True), 0.0),
                           ("goalie,    sd 10deg", replace(base, goalie=True), 10.0),
                           ("goalie,    sd 32deg", replace(base, goalie=True), 32.0)):
        for name, rates in table(cfg, args.n, sd, args.speed).items():
            print(f"{label:20s} {name:6s} " + "".join(f"{r:7.2f}" for r in rates), flush=True)


if __name__ == "__main__":
    main()
