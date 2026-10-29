#!/usr/bin/env python3
"""Pre-registered analysis: do holding / aiming rewards stop the fire-on-contact
habit from forming, when trained from scratch?

    PYTHONPATH=. python3 analysis/puck_scratch.py <experiment_dir>

Committed before any run of this screen existed. Do not edit after looking at
results.

Why: fine-tuning v8 with either reward changed nothing (puck-finetune), because
by then the habit had shut off its own exploration: "do not fire" was sampled
on ~1% of holding steps. Saved snapshots show the habit forming between 250k
steps (fire-while-holding 57%, near an untrained net's coin flip) and 3M
(100%). So the rewards have to act inside that window, from scratch.

Design: 8 starting networks (init_seed 300-307), game stream seed 17, 3M steps,
three arms run back to back per network on one machine -- control,
possession_rate=0.0008, puck_goal_speed_rate=0.01. Same starting weights in
every arm of a network.

Per run, final network, self-play under stochastic actions (the policy that
generates training data), 128 envs x 600 steps, both skaters, seeded:
  holds  agent-steps on which that skater held the puck
  fires  of those, steps on which it fired
  fire-while-holding = fires / holds

ELIGIBLE: holds >= 50. A run that barely touches the puck -- a stuck run --
says nothing about what it does when it has it, and any training change
reshuffles which networks get stuck, so it is excluded, not scored.

PRIMARY, per treatment: paired difference in fire-while-holding (treatment -
control) over networks where BOTH runs are eligible; exact two-sided
sign-flip test. Pass iff p < 0.05 AND the mean difference < 0. With fewer than
5 eligible pairs no pass is possible (the smallest p for 4 pairs is 0.125),
and the verdict is "insufficient data".

SECONDARY: mean possession time and shots on target (hockey.diagnose), shots
per run, and self-play goal% as the stuck check. A treatment that stops
firing (fire-while-holding < 0.05) while shooting less than a quarter as much
as its control is flagged as hoarding.
"""
import itertools, os, sys
import numpy as np, torch

from hockey.config import DEFAULT as C
from hockey.diagnose import diagnose
from hockey.env import VecHockeyEnv
from hockey.watch import resolve_policy
from analysis.init_survey import selfplay_goal_pct

INITS = list(range(300, 308))
ARMS = ("ctl", "poss", "aim")


def holding(pol, envs=128, steps=600, seed=11):
    torch.manual_seed(seed); np.random.seed(seed)
    env = VecHockeyEnv(num_envs=envs, cfg=C, seed=seed); obs = env.observe()
    holds = fires = 0
    for _ in range(steps):
        act = np.stack([pol.act(obs[:, a], deterministic=False) for a in (0, 1)], 1)
        for a in (0, 1):
            h = env.possessor == a
            holds += int(h.sum())
            fires += int((h & (act[:, a, 2] > C.shot_threshold)).sum())
        obs, *_ = env.step(act)
    return holds, fires


def signflip_p(d):
    d = np.asarray(d, float)
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
    return float((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


def main():
    root = sys.argv[1]
    R = {}
    print(f"{'net':>4} {'arm':<5} {'holds':>6} {'fire|hold':>9} {'hold s':>7} {'on target':>9} {'shots':>6} {'goal%':>6}")
    for k in INITS:
        for arm in ARMS:
            spec = os.path.join(root, f"{arm}-{k}", "final.pt")
            pol, _ = resolve_policy(spec)
            h, f = holding(pol)
            torch.manual_seed(0); np.random.seed(0)
            b = diagnose(pol, pol, num_envs=128, steps=600, seed=11)
            g = selfplay_goal_pct(spec)
            R[arm, k] = dict(holds=h, fwh=(f / h if h else float("nan")), hold_s=b["mean_possession_sec"],
                             on=b["shot_on_target_frac"], shots=b["shots"], goal=g)
            r = R[arm, k]
            print(f"{k:>4} {arm:<5} {h:>6} {r['fwh']:>9.3f} {r['hold_s']:>7.3f} {r['on']:>9.1%} "
                  f"{r['shots']:>6} {g:>6.1f}{'' if h >= 50 else '   (ineligible)'}", flush=True)

    print()
    for t in ("poss", "aim"):
        pairs = [k for k in INITS if R[t, k]["holds"] >= 50 and R["ctl", k]["holds"] >= 50]
        d = [R[t, k]["fwh"] - R["ctl", k]["fwh"] for k in pairs]
        if len(pairs) < 5:
            print(f"{t:<5} PRIMARY: {len(pairs)} eligible pairs -> insufficient data"); continue
        p = signflip_p(d)
        verdict = "PASS" if (p < 0.05 and np.mean(d) < 0) else "FAIL"
        print(f"{t:<5} PRIMARY: fire-while-holding {t} - ctl = {np.mean(d):+.3f} over {len(pairs)} pairs, "
              f"p = {p:.4f}  ->  {verdict}")
        for k in pairs:
            if R[t, k]["fwh"] < 0.05 and R[t, k]["shots"] < 0.25 * R["ctl", k]["shots"]:
                print(f"        hoarding flag: network {k}")
        hs = np.mean([R[t, k]["hold_s"] - R["ctl", k]["hold_s"] for k in pairs])
        on = np.mean([R[t, k]["on"] - R["ctl", k]["on"] for k in pairs])
        print(f"        secondary: possession time {hs:+.3f} s, shots on target {on:+.1%} (vs ctl, same pairs)")


if __name__ == "__main__":
    main()
