#!/usr/bin/env python3
"""Pre-registered stratum analysis for critic-slow-250k.

    PYTHONPATH=. python3 analysis/critic_slow_strata.py <experiment_dir>

Committed before any critic-slow-250k run existed. The headline verdict for
that experiment is analysis/critic_slow.py, UNCHANGED (same directory layout,
same primary, same manipulation check). This file adds the stratum breakdown
that critic-slow (1M window) computed only afterwards by hand.

Hypothesis: critic-slow's x0.1 critic over the first 1M steps helped stuck
networks (+2.9 points) and hurt healthy ones (-7.6, all four worse). The race
is settled within ~250k steps, so a 250k window should keep the benefit and
shed most of the cost.

Per stratum: mean paired change in goal% (trt - ctl) and exact sign-flip p.
Read together with critic-slow's 1M numbers, with the caveat that those ran on
a different machine:
  - previously stuck (8): benefit kept if the mean change is > 0
  - previously healthy (4): cost shed if the mean change is clearly smaller in
    magnitude than 1M's -7.6 (no significance is possible here: with 4 pairs
    the smallest two-sided p is 0.125)
These are descriptive secondaries; they do not override the primary.
"""
import os, sys
import numpy as np

from analysis.critic_slow import signflip_p, measure, STUCK_BEFORE, OK_BEFORE


def main():
    root = sys.argv[1]
    for name, ks, ref in (("previously stuck (8)", STUCK_BEFORE, "+2.9"),
                          ("previously healthy (4)", OK_BEFORE, "-7.6")):
        d = []
        for k in ks:
            gc, _ = measure(root, "ctl", k)
            gt, _ = measure(root, "trt", k)
            d.append(gt - gc)
        print(f"{name:<24} mean change {np.mean(d):+.2f} points (1M window: {ref}), "
              f"sign-flip p = {signflip_p(d):.3f}   per pair {['%+.1f' % x for x in d]}")


if __name__ == "__main__":
    main()
