#!/usr/bin/env python3
"""Pre-registered analysis for the starting-weights survey.

    PYTHONPATH=. python3 analysis/init_survey.py <survey_dir>

Committed before any survey run existed. Do not edit after looking at results;
any later analysis goes in a separate, clearly-labelled file.

Why: seed-split showed a run's fate is set by its starting weights (+7.9
points, p = 0.0014) but not what about them. This looks for a property that
predicts fate -- with one pre-registered primary, so that measuring many
properties of a few networks cannot manufacture a finding.

Design: 24 starting networks (init_seed 10, 12, 11, 17 -- the grid's known
fates -- plus 100-119), ALL on the same game stream (seed 17, the most
permissive in the grid), no shaping rates, 3M steps. Only the weights vary.

Outcome, per run: self-play goal% under stochastic actions, mean of the
2.0/2.25/2.5/2.75M snapshots, seeded probes. Stuck iff < 5%.

PRIMARY predictor: corr(V, -Phi) at the first snapshot (0.25M), in
self-play. The mechanism already measured: once the critic learns -Phi,
potential shaping contributes zero gradient, and dead runs looked like runs
whose critic won that race early (0.71 vs 0.08 at 0.25M in one pair).
Prediction: higher alignment -> stuck. Test: AUC for separating stuck from
ok, two-sided permutation p over 20k label shuffles. Pass iff p < 0.05 AND
the direction matches.

EXPLORATORY, step-zero properties of the untrained network (critic and actor
see observations normalised by statistics fitted on reset and random-play
states, approximating the first rollout):
  E1  corr(V, -Phi) at step zero
  E2  sd of the critic's initial output
  E3  fraction of saturated actor hidden units (|tanh| > 0.95)
Reported with p, flagged at a Bonferroni threshold of 0.05/3. Any hit is a
hypothesis for a confirmation run on fresh networks, not a finding.
"""
import os, sys, numpy as np, torch
from hockey.env import VecHockeyEnv
from hockey.config import DEFAULT as C
from hockey.watch import resolve_policy
from rl.ppo import PPOConfig, PPOTrainer

INITS = [10, 12, 11, 17] + list(range(100, 120))
ENV_SEED = 17
SNAPS = ("002000000", "002250000", "002500000", "002750000")


def run_dir(root, k):
    return os.path.join(root, f"init-{k}")


def selfplay_goal_pct(spec, envs=48, steps=900, seed=3):
    torch.manual_seed(seed); np.random.seed(seed)
    pol, _ = resolve_policy(spec)
    env = VecHockeyEnv(num_envs=envs, cfg=C, seed=seed); obs = env.observe()
    goals = eps = 0
    for _ in range(steps):
        act = np.stack([pol.act(obs[:, a], deterministic=False) for a in (0, 1)], 1)
        obs, _, goal, _, info = env.step(act)
        goals += int(goal.sum()); eps += int(info["episode_end"].sum())
    return 100 * goals / max(eps, 1)


def value_phi_corr(net, norm, sample_actions, envs=64, steps=400, seed=5):
    """corr(V, -Phi) over self-play states; each agent against its own potential."""
    torch.manual_seed(seed); np.random.seed(seed)
    env = VecHockeyEnv(num_envs=envs, cfg=C, seed=seed); obs = env.observe()
    V, P = [], []
    for _ in range(steps):
        phi = env._potential()
        with torch.no_grad():
            for a, sgn in ((0, 1.0), (1, -1.0)):
                V.append(net.value(torch.as_tensor(norm(obs[:, a]))).numpy())
                P.append(sgn * phi)
        obs, *_ = env.step(sample_actions(obs))
    return float(np.corrcoef(-np.concatenate(P), np.concatenate(V))[0, 1])


def step_zero(k):
    tr = PPOTrainer(PPOConfig(num_envs=4, rollout_steps=8, seed=ENV_SEED, init_seed=k), C)
    # Fit the normaliser the way the first rollout would: on reset states plus
    # a stretch of random play.
    rng = np.random.default_rng(99)
    env = VecHockeyEnv(num_envs=256, cfg=C, seed=99); obs = env.observe()
    for _ in range(60):
        tr.norm.update(obs.reshape(-1, obs.shape[-1]))
        obs, *_ = env.step(rng.uniform(-1, 1, (256, 2, 3)))
    rand = lambda o: np.random.uniform(-1, 1, (o.shape[0], 2, 3))
    e1 = value_phi_corr(tr.net, tr.norm, rand)
    x = torch.as_tensor(tr.norm(obs.reshape(-1, obs.shape[-1])))
    with torch.no_grad():
        e2 = float(tr.net.value(x).std())
        h, sat = x, []
        for layer in tr.net.actor:
            h = layer(h)
            if isinstance(layer, torch.nn.Tanh):
                sat.append((h.abs() > 0.95).float().mean().item())
    return e1, e2, float(np.mean(sat))


def snapshot_corr(spec):
    pol, _ = resolve_policy(spec)
    samp = lambda o: np.stack([pol.act(o[:, a], deterministic=False) for a in (0, 1)], 1)
    return value_phi_corr(pol.net, pol.norm, samp)


def auc(x, stuck):
    """P(a stuck run scores higher than an ok run); ties count half."""
    s, o = x[stuck], x[~stuck]
    if len(s) == 0 or len(o) == 0:
        return float("nan")
    return float(((s[:, None] > o[None, :]).sum() + 0.5 * (s[:, None] == o[None, :]).sum())
                 / (len(s) * len(o)))


def perm_p(x, stuck, n=20000):
    obs = auc(x, stuck); rng = np.random.default_rng(0)
    null = np.array([auc(x, rng.permutation(stuck)) for _ in range(n)])
    return obs, float((np.abs(null - 0.5) >= abs(obs - 0.5) - 1e-12).mean())


def main():
    root = sys.argv[1]
    rows = []
    print(f"{'init':>5} {'goal%':>6} {'fate':>6} {'P corr@0.25M':>13} {'E1 corr@0':>10} {'E2 V sd':>8} {'E3 sat':>7}")
    for k in INITS:
        d = run_dir(root, k)
        g = float(np.mean([selfplay_goal_pct(f"{d}/archive/step_{s}.pt") for s in SNAPS]))
        p = snapshot_corr(f"{d}/archive/step_000250000.pt")
        e1, e2, e3 = step_zero(k)
        rows.append((k, g, p, e1, e2, e3))
        print(f"{k:>5} {g:>6.1f} {'STUCK' if g < 5 else 'ok':>6} {p:>13.3f} {e1:>10.3f} {e2:>8.3f} {e3:>7.3f}", flush=True)
    a = np.array([r[1:] for r in rows]); stuck = a[:, 0] < 5
    print(f"\nstuck {int(stuck.sum())}/{len(stuck)}")
    names = ["PRIMARY  corr(V,-Phi) @0.25M", "E1 corr(V,-Phi) @ step 0", "E2 critic output sd", "E3 actor saturation"]
    for j, name in enumerate(names, start=1):
        A, p = perm_p(a[:, j], stuck)
        tag = ""
        if j == 1:
            tag = "PASS" if (p < 0.05 and A > 0.5) else "FAIL"
        elif p < 0.05 / 3:
            tag = "exploratory hit (needs confirmation)"
        print(f"{name:<30} AUC {A:.3f}  p = {p:.4f}  {tag}")
    known = {10: "stuck", 12: "stuck", 11: "ok", 17: "ok"}
    print("known inits on this machine: " + ", ".join(
        f"{k}: {'stuck' if g < 5 else 'ok'} (grid: {known[k]})" for k, g, *_ in rows[:4]))


if __name__ == "__main__":
    main()
