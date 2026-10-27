#!/usr/bin/env python3
"""Pre-registered causal test: does holding the critic back rescue stuck runs?

    PYTHONPATH=. python3 analysis/critic_slow.py <experiment_dir>

Committed before any run of this experiment existed. Do not edit after
looking at results.

Why: across init-survey and init-confirm, runs whose critic had absorbed the
shaping (high corr(V, -Phi)) by 250k steps tended to end up stuck -- a real
but modest correlation that missed the pre-registered bar twice. A
correlation cannot say whether the race causes the failure. This intervenes.

Design: 12 starting networks from init-confirm, game stream seed 17, no
shaping rates, 3M steps -- 8 that were stuck there (121 124 127 128 133 139
141 143) and 4 that were healthy (120 122 123 126). Each is trained twice on
one machine, back to back: control, and treatment with the critic's learning
rate x0.1 for the first 1M steps (critic_lr_scale=0.1,
critic_warmup_steps=1000000). Same starting weights in both arms, so each
pair differs only by the intervention. Measurement functions imported from
analysis/init_survey.py.

PRIMARY: paired difference in self-play goal% (treatment - control, mean of
the 2.0-2.75M snapshots), averaged over the 12 pairs. Exact two-sided
sign-flip test over all 2^12 sign patterns. Pass iff p < 0.05 AND the mean
difference > 0.

MANIPULATION CHECK (gates interpretation): paired difference in corr(V,-Phi)
at the 250k snapshot, expected NEGATIVE (treatment's critic less aligned).
Same exact test. If the treatment did not lower alignment, the experiment did
not test the race and the verdict on it is INCONCLUSIVE, whatever the primary
says.

SECONDARY: rescues (stuck in control -> ok in treatment, stuck = goal% < 5),
harms (ok in control -> stuck in treatment), and whether the control arm
reproduces each network's init-confirm fate on this machine.

Asymmetry, stated in advance: a positive primary with a passing manipulation
check means the race is a cause. A null is ambiguous, because a slowed critic
is also a worse baseline -- noisier advantages -- during the window, and that
cost could offset a real benefit.
"""
import itertools, os, sys
import numpy as np

from analysis.init_survey import selfplay_goal_pct, snapshot_corr, SNAPS

STUCK_BEFORE = [121, 124, 127, 128, 133, 139, 141, 143]
OK_BEFORE = [120, 122, 123, 126]
INITS = STUCK_BEFORE + OK_BEFORE


def signflip_p(d):
    """Exact two-sided sign-flip p for the mean of paired differences."""
    d = np.asarray(d, float); obs = abs(d.mean())
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
    return float((np.abs((signs * d).mean(1)) >= obs - 1e-12).mean())


def measure(root, arm, k):
    d = os.path.join(root, f"{arm}-{k}")
    g = float(np.mean([selfplay_goal_pct(f"{d}/archive/step_{s}.pt") for s in SNAPS]))
    c = snapshot_corr(f"{d}/archive/step_000250000.pt")
    return g, c


def main():
    root = sys.argv[1]
    G, Cr = {}, {}
    print(f"{'init':>5} {'before':>7} | {'ctl goal%':>9} {'trt goal%':>9} | {'ctl corr':>8} {'trt corr':>8}")
    for k in INITS:
        for arm in ("ctl", "trt"):
            G[arm, k], Cr[arm, k] = measure(root, arm, k)
        print(f"{k:>5} {'stuck' if k in STUCK_BEFORE else 'ok':>7} | "
              f"{G['ctl', k]:>9.1f} {G['trt', k]:>9.1f} | {Cr['ctl', k]:>8.3f} {Cr['trt', k]:>8.3f}",
              flush=True)

    dg = [G["trt", k] - G["ctl", k] for k in INITS]
    dc = [Cr["trt", k] - Cr["ctl", k] for k in INITS]
    pg, pc = signflip_p(dg), signflip_p(dc)
    manip = np.mean(dc) < 0 and pc < 0.05
    primary = np.mean(dg) > 0 and pg < 0.05

    print(f"\nMANIPULATION CHECK  alignment trt - ctl = {np.mean(dc):+.3f}, p = {pc:.4f}  "
          f"-> {'PASS' if manip else 'FAIL'}")
    print(f"PRIMARY             goal%     trt - ctl = {np.mean(dg):+.2f} points, p = {pg:.4f}  "
          f"-> {'PASS' if primary else 'FAIL'}")
    if not manip:
        verdict = "INCONCLUSIVE: the treatment did not lower alignment, so the race was not tested"
    elif primary:
        verdict = "the race is a CAUSE: holding the critic back rescues runs"
    else:
        verdict = "NULL, and ambiguous: alignment fell but outcomes did not detectably improve"
    print(f"verdict: {verdict}")

    st = lambda a, k: G[a, k] < 5
    rescues = [k for k in INITS if st("ctl", k) and not st("trt", k)]
    harms = [k for k in INITS if not st("ctl", k) and st("trt", k)]
    n_ctl_stuck = sum(st("ctl", k) for k in INITS)
    print(f"rescues: {len(rescues)}/{n_ctl_stuck} control-stuck runs {rescues}")
    print(f"harms:   {len(harms)}/{len(INITS) - n_ctl_stuck} control-ok runs {harms}")
    repro = sum((st("ctl", k)) == (k in STUCK_BEFORE) for k in INITS)
    print(f"control reproduces init-confirm fate: {repro}/{len(INITS)}")


if __name__ == "__main__":
    main()
