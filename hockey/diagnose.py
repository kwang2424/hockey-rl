#!/usr/bin/env python3
"""Behavioural dashboard: what is the policy actually *doing*?

    python -m hockey.diagnose --a runs/v0/best.pt --b runs/v0/best.pt

Return curves and even goal counts are lagging, low-resolution indicators. In
this project every plateau so far has been a reward-design error that a single
behavioural number would have exposed immediately -- the run that looked like
"not learning" for 30M steps was in fact shooting on 99% of the steps it held
the puck, which one measurement of mean possession length makes obvious.

So measure behaviour, at ~1M steps, before spending an hour on a full run.
"""

import argparse
import json

import numpy as np

from .config import Config, DEFAULT
from .env import VecHockeyEnv
from . import rink


def _on_target(cfg, pos, theta, attacking_plus_x):
    """Would a puck fired from ``pos`` along ``theta`` cross the goal mouth?

    Straight-line check, ignoring boards and the goalie-less net, which is
    exactly the question "was this shot even aimed?".
    """
    gx = np.where(attacking_plus_x, cfg.goal_line_x, -cfg.goal_line_x)
    dx = np.cos(theta)
    dy = np.sin(theta)
    toward = np.where(attacking_plus_x, dx > 1e-6, dx < -1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (gx - pos[..., 0]) / np.where(np.abs(dx) < 1e-6, np.nan, dx)
        y_at = pos[..., 1] + dy * t
    return toward & (t > 0) & (np.abs(y_at) <= cfg.goal_half_width)


def diagnose(policy_a, policy_b, num_envs=256, steps=600, seed=11,
             cfg: Config = DEFAULT, deterministic=False):
    """Behavioural statistics for team A. Sampling is stochastic by default,
    because that is the policy that actually generated the training data."""
    env = VecHockeyEnv(num_envs=num_envs, cfg=cfg, seed=seed)
    obs = env.observe()

    held = fired_while_held = 0
    shots = on_target = 0
    body_sum = blade_sum = poss_sum = 0.0
    runs, current = [], np.zeros(num_envs, dtype=np.int64)
    goals = np.zeros(2, dtype=np.int64)

    for _ in range(steps):
        act = np.zeros((num_envs, 2, 3))
        act[:, 0] = policy_a.act(obs[:, 0], deterministic=deterministic)
        act[:, 1] = policy_b.act(obs[:, 1], deterministic=deterministic)

        holding = env.possessor == 0
        will_fire = holding & (act[:, 0, 2] > cfg.shot_threshold)
        held += int(holding.sum())
        fired_while_held += int(will_fire.sum())
        if will_fire.any():
            shots += int(will_fire.sum())
            on_target += int(_on_target(cfg, env.skater_pos[will_fire, 0],
                                        env.theta[will_fire, 0],
                                        np.ones(int(will_fire.sum()), bool)).sum())

        blade = env.blade_points()[0]
        body_sum += float(np.linalg.norm(env.puck_pos - env.skater_pos[:, 0], axis=-1).mean())
        blade_sum += float(np.linalg.norm(env.puck_pos - blade[:, 0], axis=-1).mean())
        poss_sum += float(holding.mean())

        obs, _, goal, trunc, info = env.step(act)
        goals[0] += int(info["goal_a"].sum())
        goals[1] += int(info["goal_b"].sum())

        now = env.possessor == 0
        ended = (current > 0) & ~now
        runs.extend(current[ended].tolist())
        current = np.where(now, current + 1, 0)

    runs = np.array(runs) if runs else np.array([0])
    minutes = steps * num_envs * cfg.control_dt / 60.0
    return {
        "possession_frac": round(poss_sum / steps, 4),
        "mean_possession_steps": round(float(runs.mean()), 2),
        "max_possession_steps": int(runs.max()),
        "mean_possession_sec": round(float(runs.mean()) * cfg.control_dt, 3),
        "fire_while_holding": round(fired_while_held / max(held, 1), 4),
        "shots": int(shots),
        "shot_on_target_frac": round(on_target / max(shots, 1), 4),
        "mean_body_to_puck_m": round(body_sum / steps, 2),
        "mean_blade_to_puck_m": round(blade_sum / steps, 2),
        "blade_gap_m": round((body_sum - blade_sum) / steps, 2),
        "goals_for_per_min": round(float(goals[0]) / minutes, 3),
        "goals_against_per_min": round(float(goals[1]) / minutes, 3),
    }


def flags(stats):
    """Behavioural pathologies, as (name, message) pairs.

    The two shooting flags are deliberately a pair, because the reward has a
    failure mode on each side. Price a clearance too generously and the policy
    fires on contact and never carries; price it too harshly and it hoards the
    puck and never shoots. Both look like "not learning" on a return curve.
    """
    out = []
    if stats["mean_possession_sec"] < 0.25:
        out.append(("mean_possession_sec",
                    "under 0.25s: cannot carry the puck at all"))
    if stats["fire_while_holding"] > 0.9:
        out.append(("fire_while_holding",
                    "near 1.0: shoots the instant it touches the puck"))
    if stats["fire_while_holding"] < 0.02 and stats["mean_possession_sec"] > 1.0:
        out.append(("fire_while_holding",
                    "near 0.0 with long possessions: hoarding, never shoots"))
    if stats["shots"] >= 20 and stats["shot_on_target_frac"] < 0.15:
        out.append(("shot_on_target_frac", "under 0.15: shots are not aimed"))
    if stats["blade_gap_m"] < 0.30:
        out.append(("blade_gap_m",
                    "under 0.30m: closes on the puck without facing it"))
    return out


def report(stats, label=""):
    print(f"\n=== behaviour {label} ===")
    for k, v in stats.items():
        print(f"  {k:24s} {v}")
    print("  --- flags ---")
    found = flags(stats)
    for key, why in found:
        print(f"  [!] {key} = {stats[key]}: {why}")
    if not found:
        print("  none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="chase")
    ap.add_argument("--b", default="chase")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from .watch import resolve_policy
    pa, na = resolve_policy(args.a)
    pb, _ = resolve_policy(args.b)
    stats = diagnose(pa, pb, args.envs, args.steps, args.seed,
                     deterministic=args.deterministic)
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        report(stats, na)


if __name__ == "__main__":
    main()
