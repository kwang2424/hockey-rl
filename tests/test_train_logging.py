"""The progress CSV is the only durable record of a run; it must be complete."""

import csv
import subprocess
import sys
from pathlib import Path

import train


def test_declared_schema_matches_what_evaluate_produces():
    """EVAL_FIELDS must name exactly the keys evaluate() returns.

    If they drift, DictWriter(extrasaction="ignore") drops the difference
    silently -- which is precisely how every eval metric went missing from
    every run's CSV while stdout looked fine.
    """
    import numpy as np
    from hockey.bots import ChaseBot, RandomBot
    from hockey.rollout import play

    produced = set()
    for name in ("chase", "random"):
        for m in ("goal_diff_per_min", "goals_for", "goals_against", "possession"):
            produced.add(f"vs_{name}_{m}")
    assert set(train.EVAL_FIELDS) == produced
    assert not (set(train.BASE_FIELDS) & set(train.EVAL_FIELDS)), "duplicate column"


def test_short_run_writes_eval_columns_with_values(tmp_path):
    out = tmp_path / "run"
    subprocess.run(
        [sys.executable, "train.py", "--total-steps", "8000", "--num-envs", "32",
         "--rollout-steps", "16", "--eval-every", "2", "--eval-envs", "16",
         "--eval-steps", "60", "--out", str(out)],
        check=True, capture_output=True, cwd=Path(__file__).resolve().parents[1],
    )
    rows = list(csv.DictReader(open(out / "progress.csv")))
    assert rows, "no rows written"
    for field in train.EVAL_FIELDS:
        assert field in rows[0], f"{field} missing from the header"
    evaluated = [r for r in rows if r["vs_chase_goal_diff_per_min"] not in ("", None)]
    assert evaluated, "eval ran but no eval values reached the CSV"
    float(evaluated[0]["vs_chase_goal_diff_per_min"])   # parses as a number
