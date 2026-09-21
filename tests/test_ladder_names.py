"""Entrant labels. A collision silently drops an entrant from the ladder."""

import pytest

from hockey.ladder import unique_names


def test_bare_baselines_keep_their_names():
    assert unique_names(["chase", "random"]) == ["chase", "random"]


def test_single_run_keeps_short_labels():
    specs = ["random", "runs/a/archive/step_010000000.pt", "runs/a/final.pt"]
    assert unique_names(specs) == ["random", "step_010000000", "a/final"]


def test_same_milestone_from_two_runs_stays_distinct():
    """The two-arm experiment case: identical basenames, different runs.

    `policies` is keyed by name, so a collision here used to compare one arm
    against itself while the table still claimed both were present.
    """
    specs = ["runs/exp-ctl/archive/step_040000000.pt",
             "runs/exp-cap/archive/step_040000000.pt"]
    names = unique_names(specs)
    assert len(set(names)) == 2
    assert names == ["exp-ctl/step_040000000", "exp-cap/step_040000000"]


def test_every_entrant_survives_a_two_arm_ladder():
    specs = (["random"]
             + [f"runs/exp-ctl/archive/step_0{n}0000000.pt" for n in (1, 2, 3, 4)]
             + ["runs/exp-ctl/final.pt"]
             + [f"runs/exp-cap/archive/step_0{n}0000000.pt" for n in (1, 2, 3, 4)]
             + ["runs/exp-cap/final.pt"])
    assert len(set(unique_names(specs))) == len(specs)


def test_a_genuine_duplicate_is_an_error_not_a_silent_drop():
    with pytest.raises(SystemExit):
        unique_names(["runs/a/final.pt", "runs/a/final.pt"])
