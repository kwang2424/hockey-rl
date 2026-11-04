#!/usr/bin/env python3
"""Pre-registered CONFIRMATION of the init-survey post-hoc result.

    PYTHONPATH=. python3 analysis/init_confirm.py <confirm_dir>

Committed before any confirmation run existed. Do not edit after looking at
results.

What is being confirmed: in init-survey, a post-hoc test found the critic's
alignment with the shaping, corr(V, -Phi) at the 250k snapshot, tracks a
run's eventual self-play goal% (Spearman -0.544, p = 0.0075, n = 24). It was
chosen after seeing the data, so it does not count until it replicates on
networks it has never seen.

Design: 24 FRESH starting networks (init_seed 120-143), identical setup to
the survey -- same game stream (seed 17), no shaping, 3M steps, snapshots
every 250k. Measurements use the survey's own functions, imported, not
copied.

PRIMARY: Spearman(corr(V,-Phi) @250k, goal%) on the fresh 24. Pass iff the
two-sided 20k-shuffle permutation p < 0.05 AND rho < 0.

SECONDARY (reported, do not decide the verdict):
  S1  the survey's binary test (AUC, stuck = goal% < 5) on the fresh 24
  S2  OUT-OF-SAMPLE FILTER: cutoff fixed from the survey data alone before
      this file was written -- corr >= 0.460 predicts stuck (in-sample on
      the survey: 82% of stuck caught, 69% of healthy kept). Reported as
      caught / kept on the fresh 24, with no re-tuning.
  S3  pooled Spearman over survey + confirmation (n = 48), survey values read
      from init-survey.txt.

Software note: the survey ran on an earlier torch/numpy stack; this runs on
torch 2.14.1 / numpy 2.4.6 after an environment reset. Fates have survived
three hardware changes, so a replication across a stack change is a
robustness bonus, not a confound -- but it is recorded.
"""
import os, sys
import numpy as np

from analysis.init_survey import (selfplay_goal_pct, snapshot_corr, auc, perm_p,
                                  SNAPS, ENV_SEED)

INITS = list(range(120, 144))
CUTOFF = 0.460


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman(x, y):
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def spearman_p(x, y, n=20000):
    r = spearman(x, y); rng = np.random.default_rng(0)
    null = np.array([spearman(x, rng.permutation(y)) for _ in range(n)])
    return r, float((np.abs(null) >= abs(r) - 1e-12).mean())


def survey_values(path="init-survey.txt"):
    rows = [l.split() for l in open(path) if len(l.split()) == 7 and l.split()[0].isdigit()]
    return np.array([float(r[3]) for r in rows]), np.array([float(r[1]) for r in rows])


def main():
    root = sys.argv[1]
    c, g = [], []
    print(f"{'init':>5} {'goal%':>6} {'fate':>6} {'corr@250k':>10} {'filter says':>12}")
    for k in INITS:
        d = os.path.join(root, f"init-{k}")
        gk = float(np.mean([selfplay_goal_pct(f"{d}/archive/step_{s}.pt") for s in SNAPS]))
        ck = snapshot_corr(f"{d}/archive/step_000250000.pt")
        c.append(ck); g.append(gk)
        print(f"{k:>5} {gk:>6.1f} {'STUCK' if gk < 5 else 'ok':>6} {ck:>10.3f} "
              f"{'stuck' if ck >= CUTOFF else 'keep':>12}", flush=True)
    c, g = np.array(c), np.array(g); stuck = g < 5

    r, p = spearman_p(c, g)
    verdict = "PASS" if (p < 0.05 and r < 0) else "FAIL"
    print(f"\nstuck {int(stuck.sum())}/{len(stuck)}")
    print(f"PRIMARY  Spearman(corr@250k, goal%) = {r:+.3f}, p = {p:.4f}  ->  {verdict}")

    A, pa = perm_p(c, stuck)
    print(f"S1  binary AUC = {A:.3f}, p = {pa:.4f}")

    pred = c >= CUTOFF
    if stuck.any() and (~stuck).any():
        caught = (pred & stuck).sum() / stuck.sum(); kept = (~pred & ~stuck).sum() / (~stuck).sum()
        print(f"S2  filter (corr >= {CUTOFF}, fixed from survey): catches {caught:.0%} of stuck "
              f"({int((pred & stuck).sum())}/{int(stuck.sum())}), keeps {kept:.0%} of healthy "
              f"({int((~pred & ~stuck).sum())}/{int((~stuck).sum())})")
    sc, sg = survey_values()
    rp, pp = spearman_p(np.concatenate([sc, c]), np.concatenate([sg, g]))
    print(f"S3  pooled Spearman (n = {len(sc) + len(c)}) = {rp:+.3f}, p = {pp:.4f}")


if __name__ == "__main__":
    main()
