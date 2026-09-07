"""Run two policies against each other and collect frames and/or statistics."""

import numpy as np

from .config import Config, DEFAULT
from .env import VecHockeyEnv


def play(policy_a, policy_b, num_envs=1, steps=600, seed=0, cfg: Config = DEFAULT,
         collect_frames=False, renderer=None, env_idx=0, deterministic=True):
    """Play ``policy_a`` (team A / +x) against ``policy_b`` (team B / -x).

    Returns a dict of aggregate statistics, and the rendered frames for
    ``env_idx`` when ``collect_frames`` is set.
    """
    env = VecHockeyEnv(num_envs=num_envs, cfg=cfg, seed=seed)
    obs = env.observe()

    goals = np.zeros(2, dtype=np.int64)
    shots = np.zeros(2, dtype=np.int64)
    possession = np.zeros(2, dtype=np.float64)
    episodes = 0
    ep_lengths = []
    frames = []
    score = [0, 0]

    for _ in range(steps):
        act = np.zeros((num_envs, 2, 3))
        act[:, 0] = policy_a.act(obs[:, 0], deterministic=deterministic)
        act[:, 1] = policy_b.act(obs[:, 1], deterministic=deterministic)

        if collect_frames and renderer is not None:
            frames.append(renderer.frame(env.state_snapshot(), env_idx, score=tuple(score)))

        obs, rew, goal, trunc, info = env.step(act)

        goals[0] += int(info["goal_a"].sum())
        goals[1] += int(info["goal_b"].sum())
        ended = info["episode_end"]
        if np.any(ended):
            episodes += int(ended.sum())
            shots += info["ep_shots"][ended].sum(axis=0)
            possession += info["ep_possession"][ended].sum(axis=0)
            ep_lengths.extend(info["ep_len"][ended].tolist())
        if collect_frames:
            score[0] += int(info["goal_a"][env_idx])
            score[1] += int(info["goal_b"][env_idx])

    total_time = max(possession.sum(), 1e-9)
    minutes = steps * num_envs * cfg.control_dt / 60.0
    stats = {
        "goals_a": int(goals[0]),
        "goals_b": int(goals[1]),
        "goal_diff_per_min": float(goals[0] - goals[1]) / minutes,
        "goals_a_per_min": float(goals[0]) / minutes,
        "goals_b_per_min": float(goals[1]) / minutes,
        "shots_a": int(shots[0]),
        "shots_b": int(shots[1]),
        "possession_a_frac": float(possession[0] / total_time),
        "episodes": episodes,
        "mean_ep_len": float(np.mean(ep_lengths)) if ep_lengths else float("nan"),
        "sim_minutes": minutes,
    }
    return stats, frames
