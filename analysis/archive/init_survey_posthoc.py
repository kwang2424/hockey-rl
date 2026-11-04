#!/usr/bin/env python3
"""POST-HOC, written after seeing the init_survey results. Not pre-registered;
it cannot change that survey's verdict (primary FAILED, p = 0.063).

Motivation, stated after the fact: 8 of 24 runs landed within 1.5 points of
the 5% stuck cutoff, so the binary split discards information. This
re-tests the same four predictors against goal% as a continuous outcome
(Spearman rank correlation, two-sided permutation p over 20k shuffles).

    python3 analysis/init_survey_posthoc.py <survey_result.log>
"""
import sys, numpy as np

rows = []
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) == 7 and p[0].isdigit():
        rows.append([float(p[1])] + [float(v) for v in p[3:7]])
a = np.array(rows)

def rank(x):
    return np.argsort(np.argsort(x)).astype(float)

def spearman(x, y):
    return float(np.corrcoef(rank(x), rank(y))[0, 1])

rng = np.random.default_rng(0)
names = ["corr(V,-Phi) @0.25M (was primary)", "E1 corr(V,-Phi) @ step 0",
         "E2 critic output sd", "E3 actor saturation"]
print(f"n = {len(a)} runs; outcome = self-play goal%, continuous")
for j, n in enumerate(names, start=1):
    r = spearman(a[:, j], a[:, 0])
    null = np.array([spearman(a[:, j], rng.permutation(a[:, 0])) for _ in range(20000)])
    p = float((np.abs(null) >= abs(r) - 1e-12).mean())
    print(f"  {n:<36} Spearman {r:+.3f}   p = {p:.4f}")
