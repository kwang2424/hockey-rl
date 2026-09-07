#!/usr/bin/env python3
"""Play against a trained agent yourself.

    python -m hockey.play --opponent runs/v0/best.pt

Controls
    W / S or Up / Down     accelerate / brake and skate backwards
    A / D or Left / Right  turn
    Space                  shoot (only does anything while you have the puck)
    R                      reset the faceoff
    Tab                    let the agent drive your side too, and just watch
    Esc                    quit

You are red and attack the right-hand net. This is the single most useful
debugging tool in the project: thirty seconds of playing against a policy
tells you more about what it has learned than any curve.
"""

import argparse
import sys

import numpy as np

from .config import DEFAULT
from .env import VecHockeyEnv
from .render import Renderer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="chase", help="path.pt | chase | random | still")
    ap.add_argument("--scale", type=float, default=15.0)
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()

    try:
        import pygame
    except ImportError:
        sys.exit("pygame is required for interactive play:  pip install pygame\n"
                 "(to watch without a display instead, use: python -m hockey.watch)")

    from .watch import resolve_policy
    opponent, opp_name = resolve_policy(args.opponent)

    env = VecHockeyEnv(num_envs=1, cfg=DEFAULT, seed=np.random.randint(10**6))
    r = Renderer(DEFAULT, scale=args.scale)

    pygame.init()
    screen = pygame.display.set_mode((r.w, r.h))
    pygame.display.set_caption(f"hockey-rl  --  you (red) vs {opp_name} (blue)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    obs = env.observe()
    score = [0, 0]
    autopilot = None
    running = True

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    env._reset_idx(np.array([0]))
                    obs = env.observe()
                elif ev.key == pygame.K_TAB:
                    autopilot = None if autopilot else resolve_policy(args.opponent)[0]

        keys = pygame.key.get_pressed()
        if autopilot is not None:
            mine = autopilot.act(obs[:, 0], deterministic=not args.stochastic)[0]
        else:
            fwd = (keys[pygame.K_w] or keys[pygame.K_UP]) - (keys[pygame.K_s] or keys[pygame.K_DOWN])
            # Screen y grows downward, so "left on screen" is +theta in world space.
            turn = (keys[pygame.K_a] or keys[pygame.K_LEFT]) - (keys[pygame.K_d] or keys[pygame.K_RIGHT])
            shoot = 1.0 if keys[pygame.K_SPACE] else -1.0
            mine = np.array([float(fwd), float(turn), shoot])

        act = np.zeros((1, 2, 3))
        act[0, 0] = mine
        act[0, 1] = opponent.act(obs[:, 1], deterministic=not args.stochastic)[0]

        frame = r.frame(env.state_snapshot(), 0, score=tuple(score))
        surf = pygame.image.frombuffer(frame.tobytes(), frame.size, "RGB")
        screen.blit(surf, (0, 0))

        owner = int(env.possessor[0])
        hud = "PUCK" if owner == 0 else ("opponent has it" if owner == 1 else "loose")
        speed = float(np.linalg.norm(env.skater_vel[0, 0]))
        screen.blit(font.render(f"{hud}   {speed:4.1f} m/s   [Tab] autopilot  [R] reset",
                                True, (60, 60, 70)), (10, r.h - 24))
        pygame.display.flip()

        obs, _, goal, trunc, info = env.step(act)
        score[0] += int(info["goal_a"][0])
        score[1] += int(info["goal_b"][0])
        clock.tick(int(round(1.0 / DEFAULT.control_dt)))

    pygame.quit()


if __name__ == "__main__":
    main()
