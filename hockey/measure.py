"""Standard measurements, shared by every experiment's analysis script.

Each one is seeded, so the same networks always give the same numbers. They
play the game the Config describes (the default is the solid-net goalie game).

    selfplay(p)              goals, shots, goals per shot and per episode
    h2h(a, b)                head to head, both ends: (goals for a, for b)
    behaviour(p)             hockey.diagnose stats in self-play
    shot_geometry(p)         shot distance, heading error, possessions ending in a shot
    anchored_rating(p, ...)  hockey.rating over a fixed set of opponents
"""
import numpy as np
import torch

from .config import Config, DEFAULT
from .diagnose import diagnose
from .env import VecHockeyEnv
from .rating import rating
from .rollout import play


def selfplay(p, cfg: Config = DEFAULT, envs=128, steps=600, seed=11, deterministic=False):
    torch.manual_seed(0); np.random.seed(0)
    s, _ = play(p, p, num_envs=envs, steps=steps, seed=seed, cfg=cfg, deterministic=deterministic)
    goals, shots = s["goals_a"] + s["goals_b"], s["shots_a"] + s["shots_b"]
    return dict(goals=goals, shots=shots, gps=goals / max(shots, 1),
                goals_per_ep=goals / max(s["episodes"], 1))


def h2h(pa, pb, cfg: Config = DEFAULT, envs=96, steps=400, seed=0):
    s1, _ = play(pa, pb, num_envs=envs, steps=steps, seed=seed, cfg=cfg)
    s2, _ = play(pb, pa, num_envs=envs, steps=steps, seed=seed + 1, cfg=cfg)
    return s1["goals_a"] + s2["goals_b"], s1["goals_b"] + s2["goals_a"]


def behaviour(p, cfg: Config = DEFAULT):
    torch.manual_seed(0); np.random.seed(0)
    return diagnose(p, p, num_envs=128, steps=600, seed=11, cfg=cfg)


def shot_geometry(pa, pb, cfg: Config = DEFAULT, envs=128, steps=900, seed=11):
    """Team A's shots: heading error at release (vs the line to the goal
    centre), distance to goal, and the share of possessions ending in a shot."""
    torch.manual_seed(0); np.random.seed(0)
    env = VecHockeyEnv(num_envs=envs, cfg=cfg, seed=seed)
    obs = env.observe()
    goal = env._goal_centers[0]
    err, dist = [], []
    shot_end = lost_end = 0
    for _ in range(steps):
        act = np.stack([pa.act(obs[:, 0], deterministic=False),
                        pb.act(obs[:, 1], deterministic=False)], 1)
        held = env.possessor == 0
        pos, th = env.skater_pos[:, 0].copy(), env.theta[:, 0].copy()
        obs, *_ = env.step(act)
        f = env.fired[:, 0]
        if f.any():
            to_g = goal - pos[f]
            bearing = np.arctan2(to_g[:, 1], to_g[:, 0])
            e = np.abs((th[f] - bearing + np.pi) % (2 * np.pi) - np.pi)
            err += np.degrees(e).tolist()
            dist += np.linalg.norm(to_g, axis=1).tolist()
        shot_end += int(f.sum())
        lost_end += int((held & ~f & (env.possessor != 0)).sum())
    q = lambda a, pc: float(np.percentile(a, pc)) if len(a) else float("nan")
    return dict(n=len(err), med=q(err, 50), dist=q(dist, 50),
                shot_share=shot_end / max(shot_end + lost_end, 1))


def anchored_rating(p, anchors, cfg: Config = DEFAULT, envs=128, steps=600, seed=12345):
    """`anchors` is a list of (name, policy). Returns (rating, {name: (for, against)})."""
    counts = {}
    for name, opp in anchors:
        s, _ = play(p, opp, num_envs=envs, steps=steps, seed=seed, cfg=cfg)
        counts[name] = (s["goals_a"], s["goals_b"])
    return rating(counts), counts
