#!/usr/bin/env python3
"""Train the v0 1v1 hockey policy by self-play.

    python train.py --total-steps 3000000 --out runs/v0

Progress is measured against the scripted ChaseBot, not against the return
curve. Self-play return is close to meaningless on its own -- a policy getting
better and a policy getting worse both hover around zero, because it is always
playing something exactly as good as itself.
"""

import argparse
import csv
import json
import os
import time

import numpy as np

from hockey.config import DEFAULT
from hockey.bots import ChaseBot, RandomBot
from hockey.rollout import play
from rl.ppo import PPOTrainer, PPOConfig


def evaluate(trainer, steps, envs, seed=12345):
    """Objective progress check: play the learner against fixed baselines."""
    pol = trainer.policy()
    out = {}
    for name, opp in (("chase", ChaseBot()), ("random", RandomBot(seed=seed))):
        s, _ = play(pol, opp, num_envs=envs, steps=steps, seed=seed)
        out[f"vs_{name}_goal_diff_per_min"] = round(s["goal_diff_per_min"], 3)
        out[f"vs_{name}_goals_for"] = s["goals_a"]
        out[f"vs_{name}_goals_against"] = s["goals_b"]
        out[f"vs_{name}_possession"] = round(s["possession_a_frac"], 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=3_000_000)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--rollout-steps", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy-coef", type=float, default=0.004)
    ap.add_argument("--pool-prob", type=float, default=0.35)
    ap.add_argument("--curriculum-start", type=float, default=0.75,
                    help="initial fraction of resets that start with the puck on a stick")
    ap.add_argument("--curriculum-end", type=float, default=0.05)
    ap.add_argument("--curriculum-frac", type=float, default=0.6,
                    help="fraction of training spent annealing the curriculum")
    ap.add_argument("--eval-every", type=int, default=15)
    ap.add_argument("--eval-steps", type=int, default=300)
    ap.add_argument("--eval-envs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/v0")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    p = PPOConfig(
        num_envs=args.num_envs, rollout_steps=args.rollout_steps,
        total_steps=args.total_steps, lr=args.lr, entropy_coef=args.entropy_coef,
        pool_prob=args.pool_prob, eval_every=args.eval_every,
        eval_steps=args.eval_steps, eval_envs=args.eval_envs,
        curriculum_start=args.curriculum_start, curriculum_end=args.curriculum_end,
        curriculum_frac=args.curriculum_frac,
        seed=args.seed, device=args.device, out_dir=args.out,
    )
    os.makedirs(args.out, exist_ok=True)
    trainer = PPOTrainer(p, DEFAULT)

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"ppo": p.__dict__, "env": DEFAULT.to_dict()}, f, indent=2, default=str)

    csv_path = os.path.join(args.out, "progress.csv")
    writer, csv_file, t0 = None, None, time.time()
    per_update = p.num_envs * p.rollout_steps
    n_updates = max(1, args.total_steps // per_update)
    best = -1e9

    print(f"[train] {n_updates} updates x {per_update:,} steps = {n_updates*per_update:,} total")
    for update in range(1, n_updates + 1):
        trainer.update = update
        if p.anneal_lr:
            frac = 1.0 - (update - 1) / n_updates
            for g in trainer.opt.param_groups:
                g["lr"] = frac * p.lr

        curriculum = trainer.set_curriculum((update - 1) / n_updates)
        batch, rstats = trainer.collect()
        rstats["curriculum"] = round(curriculum, 4)
        lstats = trainer.learn_on(batch)

        if update % p.snapshot_every == 0:
            trainer._snapshot()

        row = {"update": update, "step": trainer.global_step,
               "elapsed_s": round(time.time() - t0, 1),
               "sps": int(trainer.global_step / max(time.time() - t0, 1e-9)),
               **rstats, **{k: v for k, v in lstats.items() if k != "n_train"}}

        if update % p.eval_every == 0 or update == n_updates:
            row.update(evaluate(trainer, p.eval_steps, p.eval_envs))
            score = row["vs_chase_goal_diff_per_min"]
            if score > best:
                best = score
                trainer.save(os.path.join(args.out, "best.pt"))
            trainer.save(os.path.join(args.out, "latest.pt"))
            print(f"[{update:4d}/{n_updates}] step={trainer.global_step:>9,} "
                  f"sps={row['sps']:>6,} ent={row.get('entropy',0):.2f} "
                  f"vs_chase={row['vs_chase_goal_diff_per_min']:+.2f}/min "
                  f"({row['vs_chase_goals_for']}-{row['vs_chase_goals_against']}) "
                  f"vs_random={row['vs_random_goal_diff_per_min']:+.2f}/min "
                  f"poss={row['vs_chase_possession']:.2f} "
                  f"curr={row['curriculum']:.2f}", flush=True)

        if writer is None:
            csv_file = open(csv_path, "w", newline="")
            writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()), extrasaction="ignore")
            writer.writeheader()
        writer.writerow(row)
        csv_file.flush()

    trainer.save(os.path.join(args.out, "final.pt"))
    if csv_file:
        csv_file.close()
    print(f"[train] done in {time.time()-t0:.0f}s -> {args.out}/  (best vs_chase {best:+.2f}/min)")


if __name__ == "__main__":
    main()
