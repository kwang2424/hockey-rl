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

    Asserted against a real evaluate() call rather than a hand-copied list of
    column names, so that adding or renaming an anchor cannot pass the test
    while silently dropping that anchor's columns from the CSV.
    """
    from hockey.config import DEFAULT
    from rl.ppo import PPOConfig, PPOTrainer

    specs = ("chase", "checkpoints/v2-control-gated-reward.pt")
    trainer = PPOTrainer(PPOConfig(num_envs=4, rollout_steps=8), DEFAULT)
    anchors = train.resolve_anchors(specs)
    produced = train.evaluate(trainer, steps=20, envs=4, anchors=anchors)

    assert set(train.eval_fields(specs)) == set(produced)
    assert not (set(train.BASE_FIELDS) & set(train.EVAL_FIELDS)), "duplicate column"


def test_rating_is_recorded_and_selects_best(tmp_path):
    """best.pt is chosen by eval_rating, not by one anchor's goal difference."""
    out = tmp_path / "run"
    subprocess.run(
        [sys.executable, "train.py", "--total-steps", "8000", "--num-envs", "32",
         "--rollout-steps", "16", "--eval-every", "2", "--eval-envs", "16",
         "--eval-steps", "60", "--eval-anchors", "chase,random", "--out", str(out)],
        check=True, capture_output=True, cwd=Path(__file__).resolve().parents[1],
    )
    rows = [r for r in csv.DictReader(open(out / "progress.csv"))
            if r["eval_rating"] not in ("", None)]
    assert rows, "no eval rows"
    import math
    from hockey.rating import rating
    r = rows[0]
    expected = rating({k: (int(r[f"vs_{k}_goals_for"]), int(r[f"vs_{k}_goals_against"]))
                       for k in ("chase", "random")})
    assert math.isclose(float(r["eval_rating"]), expected, abs_tol=1e-3)
    assert (out / "best.pt").exists()


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


def _training_cells(path):
    cols = ("mean_reward", "entropy", "pg_loss", "v_loss", "grad_norm", "goals_per_ep")
    return [tuple(r[c] for c in cols) for r in csv.DictReader(open(path))]


def _short_run(out, anchors):
    subprocess.run(
        [sys.executable, "train.py", "--total-steps", "12000", "--num-envs", "32",
         "--rollout-steps", "16", "--eval-every", "5", "--eval-envs", "16",
         "--eval-steps", "40", "--seed", "7", "--eval-anchors", anchors,
         "--out", str(out)],
        check=True, capture_output=True, cwd=Path(__file__).resolve().parents[1],
    )
    return _training_cells(out / "progress.csv")


def test_evaluation_does_not_perturb_training(tmp_path):
    """Measuring must not change what is measured.

    Building an anchor policy constructs an ActorCritic, and its layer
    initialisation draws from the global torch RNG -- so adding an anchor used
    to shift the training action-sampling stream and silently turn a
    one-change experiment into a difference of seeds as well. Same seed, two
    anchor sets, identical training trace.
    """
    two = _short_run(tmp_path / "two", "chase,random")
    four = _short_run(tmp_path / "four",
                      "chase,random,checkpoints/v2-control-gated-reward.pt,"
                      "checkpoints/v6-100m-compute.pt")
    assert two and len(two) == len(four)
    assert two == four
