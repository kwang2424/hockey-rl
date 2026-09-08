#!/usr/bin/env python3
"""Scripted manoeuvres: does a skater move the way a person would expect?

    python -m hockey.drills                 # run them all, print a report
    python -m hockey.drills --gif out.gif   # and render them with path trails

This is play-testing without a keyboard. Each drill drives a skater with a
fixed sequence of control inputs -- exactly what `hockey.play` sends when you
hold a key -- and measures what came out. It answers a question the policy
metrics cannot: is the *vehicle* sound, independently of whether the policy has
learned to drive it?

That distinction has mattered here. "The agent never holds the puck" could mean
the carry mechanic is broken or that the policy never learned to use it; the
carry drill separates those in about a second.
"""

import argparse

import numpy as np

from .config import Config, DEFAULT
from .env import VecHockeyEnv, ACT_DIM

# Control inputs are [forward, turn, shoot], each in [-1, 1] -- the same three
# numbers the keyboard produces in hockey.play.
FWD = [1.0, 0.0, -1.0]
COAST = [0.0, 0.0, -1.0]
BRAKE = [-1.0, 0.0, -1.0]
LEFT = [1.0, 1.0, -1.0]
RIGHT = [1.0, -1.0, -1.0]
PIVOT_L = [0.0, 1.0, -1.0]
SHOOT = [0.0, 0.0, 1.0]


def _script(*segments):
    """(action, n_steps) pairs -> a flat list of per-step actions."""
    out = []
    for action, n in segments:
        out.extend([list(action)] * n)
    return out


DRILLS = {
    "accelerate": _script((FWD, 90)),
    "hard_left_at_speed": _script((FWD, 45), (LEFT, 90)),
    "pivot_in_place": _script((PIVOT_L, 40)),
    "stop_from_top_speed": _script((FWD, 90), (BRAKE, 60)),
    "figure_eight": _script((FWD, 20), (LEFT, 55), (RIGHT, 55), (LEFT, 55), (RIGHT, 55)),
    "carry_the_puck": _script((FWD, 120)),
    "carry_and_turn": _script((FWD, 60), (LEFT, 60), (FWD, 40)),
}


def run(name, cfg: Config = DEFAULT, start_pos=None, start_theta=0.0,
        puck_offset=None, collect=True):
    """Run one drill on a lone skater. Returns a dict of measurements."""
    env = VecHockeyEnv(num_envs=1, cfg=cfg, seed=0)
    env.skater_pos[0] = [start_pos or [-14.0, -6.0], [-28.0, 12.0]]  # partner parked away
    env.theta[0] = [start_theta, 0.0]
    env.omega[:] = 0.0
    env.skater_vel[:] = 0.0
    # Puck either right on the blade (carry drills) or safely out of the way.
    if puck_offset is None:
        env.puck_pos[:] = [0.0, -12.5]
    else:
        heading = np.array([np.cos(start_theta), np.sin(start_theta)])
        env.puck_pos[:] = np.array(start_pos or [-14.0, -6.0]) + heading * puck_offset
    env.puck_vel[:] = 0.0
    env.possessor[:] = -1

    path, puck_path, speeds, headings, held = [], [], [], [], []
    for act in DRILLS[name]:
        a = np.zeros((1, 2, ACT_DIM))
        a[0, 0] = act
        env.step(a)
        path.append(env.skater_pos[0, 0].copy())
        puck_path.append(env.puck_pos[0].copy())
        speeds.append(float(np.linalg.norm(env.skater_vel[0, 0])))
        headings.append(float(env.theta[0, 0]))
        held.append(env.possessor[0] == 0)

    path = np.array(path); puck_path = np.array(puck_path)
    speeds = np.array(speeds); held = np.array(held)
    swept = np.abs(np.unwrap(headings) - headings[0])
    steps = len(path)

    # Turn radius as v / omega while actually skating and actually turning.
    # Taking the tightest curvature anywhere on the path is misleading: the
    # tightest point is wherever the skater was slowest, which says nothing
    # about how well it carves at speed.
    omegas = np.abs(np.diff(np.unwrap(headings))) / cfg.control_dt
    fast = speeds[1:] > 5.0
    turning = omegas > 0.2
    sel = fast & turning
    radius = float(np.median(speeds[1:][sel] / omegas[sel])) if sel.any() else float("nan")

    return {
        "steps": steps,
        "speed_retained": round(float(speeds[-1] / max(speeds.max(), 1e-9)), 3),
        "duration_s": round(steps * cfg.control_dt, 2),
        "top_speed": round(float(speeds.max()), 2),
        "final_speed": round(float(speeds[-1]), 2),
        "distance_m": round(float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()), 1),
        "heading_swept_deg": round(float(np.degrees(swept.max())), 1),
        "turn_radius_at_speed_m": round(radius, 2) if radius == radius else None,
        "possession_frac": round(float(held.mean()), 3),
        "longest_hold_s": round(float(_longest_run(held) * cfg.control_dt), 2),
        "_path": path, "_puck": puck_path,
    }


def _longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


CHECKS = {
    "accelerate": lambda r, c: (r["top_speed"] > 0.9 * c.max_speed,
                                f"reached only {r['top_speed']} of {c.max_speed} m/s"),
    # A hard turn must be a carve, not a brake: the skater kept 84% of its
    # speed here only after the turn rate was capped by the grip budget.
    "hard_left_at_speed": lambda r, c: (r["heading_swept_deg"] > 180 and r["speed_retained"] > 0.7,
                                        f"swept {r['heading_swept_deg']} deg but kept only "
                                        f"{r['speed_retained']:.0%} of its speed"),
    "pivot_in_place": lambda r, c: (r["heading_swept_deg"] > 180 and r["distance_m"] < 1.0,
                                    f"swept {r['heading_swept_deg']} deg over {r['distance_m']} m"),
    "stop_from_top_speed": lambda r, c: (r["final_speed"] < 3.0,
                                         f"still moving at {r['final_speed']} m/s"),
    "figure_eight": lambda r, c: (r["heading_swept_deg"] > 180,
                                  f"swept only {r['heading_swept_deg']} deg"),
    "carry_the_puck": lambda r, c: (r["longest_hold_s"] > 2.0,
                                    f"longest hold only {r['longest_hold_s']}s"),
    "carry_and_turn": lambda r, c: (r["longest_hold_s"] > 1.5,
                                    f"lost the puck after {r['longest_hold_s']}s"),
}

CARRY_DRILLS = {"carry_the_puck", "carry_and_turn"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", default=None, help="render the drills to this GIF")
    ap.add_argument("--only", default=None, help="run a single drill by name")
    args = ap.parse_args()

    names = [args.only] if args.only else list(DRILLS)
    frames = []
    failures = 0

    for name in names:
        offset = DEFAULT.blade_offset if name in CARRY_DRILLS else None
        r = run(name, puck_offset=offset)
        ok, why = CHECKS[name](r, DEFAULT)
        failures += (not ok)
        flag = "ok  " if ok else "FAIL"
        print(f"[{flag}] {name:20s} {r['duration_s']:5.2f}s  top {r['top_speed']:5.2f} m/s  "
              f"swept {r['heading_swept_deg']:6.1f} deg  radius {r['turn_radius_at_speed_m']}  "
              f"hold {r['longest_hold_s']:4.2f}s" + ("" if ok else f"   <- {why}"))

        if args.gif:
            from .render import Renderer
            rend = Renderer(DEFAULT, scale=13.0)
            env = VecHockeyEnv(num_envs=1, cfg=DEFAULT, seed=0)
            for i in range(len(r["_path"])):
                env.skater_pos[0, 0] = r["_path"][i]
                env.skater_pos[0, 1] = [-28.0, 12.0]
                env.puck_pos[0] = r["_puck"][i]
                if i:
                    step = r["_path"][i] - r["_path"][i - 1]
                    if np.linalg.norm(step) > 1e-6:
                        env.theta[0, 0] = np.arctan2(step[1], step[0])
                trails = [(r["_path"][: i + 1], (230, 120, 120)),
                          (r["_puck"][: i + 1], (90, 90, 100))]
                frames.append(rend.frame(env.state_snapshot(), 0, label=name, trails=trails))

    if args.gif and frames:
        from .render import save_gif
        save_gif(frames[::2], args.gif, fps=20)
        print(f"\nwrote {args.gif} ({len(frames[::2])} frames)")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
