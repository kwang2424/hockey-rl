#!/usr/bin/env python3
"""Record a game to an animated GIF.

    python -m hockey.watch --a runs/v0/best.pt --b chase --out game.gif

Works headless -- no display, no game engine. This is the fastest way to find
out what a policy has actually learned, which is rarely what the reward curve
suggests.
"""

import argparse

import numpy as np

from .config import DEFAULT
from .bots import ChaseBot, RandomBot, StandStillBot
from .render import Renderer, save_gif
from .rollout import play


def resolve_policy(spec, device="cpu"):
    """'chase' | 'random' | 'still' | a path to a .pt checkpoint."""
    if spec == "chase":
        return ChaseBot(), "chase"
    if spec == "random":
        return RandomBot(seed=0), "random"
    if spec == "still":
        return StandStillBot(), "still"
    from rl.ppo import load_policy
    pol, ck = load_policy(spec, device=device)
    return pol, f"{spec}@{ck['global_step']:,}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="chase", help="team A policy (red, attacks +x)")
    ap.add_argument("--b", default="chase", help="team B policy (blue, attacks -x)")
    ap.add_argument("--out", default="game.gif")
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=11.0)
    ap.add_argument("--every", type=int, default=2, help="keep 1 frame in N")
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()

    pa, na = resolve_policy(args.a)
    pb, nb = resolve_policy(args.b)
    r = Renderer(DEFAULT, scale=args.scale)

    stats, frames = play(pa, pb, num_envs=1, steps=args.steps, seed=args.seed,
                         collect_frames=True, renderer=r,
                         deterministic=not args.stochastic)
    frames = frames[::args.every]
    fps = max(1, int(round(1.0 / DEFAULT.control_dt / args.every)))
    save_gif(frames, args.out, fps=fps)
    print(f"{na} (red) vs {nb} (blue): {stats['goals_a']}-{stats['goals_b']} "
          f"| poss A {stats['possession_a_frac']:.2f} "
          f"| shots {stats['shots_a']}-{stats['shots_b']}")
    print(f"wrote {args.out}  ({len(frames)} frames @ {fps} fps)")


if __name__ == "__main__":
    main()
