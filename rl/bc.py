#!/usr/bin/env python3
"""Behaviour cloning: start a network by copying a scripted player.

    PYTHONPATH=. python3 -m rl.bc --out checkpoints/bc-sniper.pt

Why: in the solid-net goalie game, 28 of 28 reinforcement-learning runs never
learned to score (EXPERIMENTS.md, goalie-v9 and escape). Scoring needs a long
chain -- win the puck, carry it to ~10 m, line up on a corner, shoot -- and
random exploration never completes it. SniperBot does exactly that chain, so
the network first learns to imitate it, then PPO takes over from there
("kickstarting", as AlphaStar started from human replays).

What it does:
  1. Records SniperBot against itself. The moves actually played get Gaussian
     noise on skating and turning (and an occasional flipped shoot bit), but
     every state is labelled with SniperBot's clean action -- so the copy also
     sees off-script states and what the script would do from them, rather
     than only the script's own narrow trajectory.
  2. Fits the observation normaliser to the recorded states, the actor's mean
     to SniperBot's actions (squared error), and the critic to the discounted
     return each skater actually got (the env's own reward, shaping
     included), so PPO does not start from an untrained critic.
  3. Saves a normal trainer checkpoint, usable with --init-from and
     load_policy.

The policy's spread (log_std) is set, not learned: max likelihood would
shrink it to the tiny imitation error and leave PPO nothing to explore with.
"""
import argparse
from dataclasses import replace

import numpy as np
import torch

from hockey.bots import SniperBot
from hockey.config import DEFAULT
from hockey.env import VecHockeyEnv, ACT_DIM
from rl.ppo import PPOTrainer, PPOConfig

GAME = replace(DEFAULT, solid_net=True, goalie=True)


def record(cfg=GAME, envs=256, steps=3000, move_noise=0.3, flip_shoot=0.02, seed=0):
    """(obs, clean_action, return) for both skaters, flattened."""
    rng = np.random.default_rng(seed)
    bot = SniperBot(cfg, noise=0.0)
    env = VecHockeyEnv(num_envs=envs, cfg=cfg, seed=seed)
    obs = env.observe()
    O = np.zeros((steps, envs, 2, env.obs_dim), dtype=np.float32)
    A = np.zeros((steps, envs, 2, ACT_DIM), dtype=np.float32)
    R = np.zeros((steps, envs, 2), dtype=np.float32)
    D = np.zeros((steps, envs), dtype=bool)
    for t in range(steps):
        clean = bot.act(obs)
        played = clean.copy()
        played[..., :2] += rng.normal(0.0, move_noise, played[..., :2].shape)
        flip = rng.random(played.shape[:-1]) < flip_shoot
        played[..., 2] = np.where(flip, -played[..., 2], played[..., 2])
        O[t], A[t] = obs, clean
        obs, r, goal, trunc, info = env.step(np.clip(played, -1.0, 1.0))
        R[t] = r
        D[t] = info["episode_end"]
    # Discounted return-to-go, cut at episode ends. A truncated episode's tail
    # is cut to zero rather than bootstrapped; with gamma 0.995 that is a small
    # bias, and only for the last few hundred steps of an episode.
    G = np.zeros_like(R)
    run = np.zeros((envs, 2), dtype=np.float32)
    for t in range(steps - 1, -1, -1):
        run = np.where(D[t][:, None], 0.0, run)
        run = R[t] + cfg.gamma * run
        G[t] = run
    # The last ~1000 steps of each env have an unfinished return; drop them.
    keep = steps - 1000
    return (O[:keep].reshape(-1, env.obs_dim), A[:keep].reshape(-1, ACT_DIM),
            G[:keep].reshape(-1))


def train(obs, act, ret, cfg=GAME, epochs=30, batch=4096, lr=1e-3, log_std=-1.0, seed=0):
    torch.manual_seed(seed)
    t = PPOTrainer(PPOConfig(num_envs=8, rollout_steps=8, seed=seed), cfg)
    t.norm.update(obs)
    x = torch.as_tensor(t.norm(obs))
    a = torch.as_tensor(act)
    g = torch.as_tensor(ret, dtype=torch.float32)
    with torch.no_grad():
        t.net.log_std.fill_(log_std)
    params = list(t.net.actor.parameters()) + list(t.net.critic.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    n = len(x)
    gen = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen)
        la = lv = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            mean = t.net.actor(x[b])
            loss_a = ((mean - a[b]) ** 2).mean()
            loss_v = ((t.net.value(x[b]) - g[b]) ** 2).mean()
            opt.zero_grad()
            (loss_a + 0.5 * loss_v).backward()
            opt.step()
            la += loss_a.item() * len(b); lv += loss_v.item() * len(b)
        print(f"[bc] epoch {ep + 1}/{epochs}  action mse {la / n:.4f}  value mse {lv / n:.5f}", flush=True)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/bc-sniper.pt")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--log-std", type=float, default=-1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    obs, act, ret = record(envs=args.envs, steps=args.steps, seed=args.seed)
    print(f"[bc] recorded {len(obs):,} labelled states", flush=True)
    t = train(obs, act, ret, epochs=args.epochs, log_std=args.log_std, seed=args.seed)
    t.save(args.out)
    print(f"[bc] wrote {args.out}")


if __name__ == "__main__":
    main()
