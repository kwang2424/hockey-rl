"""Milestone selection for the seed-spread measurement."""

import os

import pytest

from hockey.seedspread import milestones


def _run(tmp, name, steps, final=True):
    d = tmp / name
    (d / "archive").mkdir(parents=True)
    for s in steps:
        (d / "archive" / f"step_{s:09d}.pt").write_text("x")
    if final:
        (d / "final.pt").write_text("x")
    return str(d)


def test_uses_only_milestones_present_in_every_run(tmp_path):
    """A pairing is like-for-like or it is not a pairing. A run that died
    early contributes fewer rows rather than being compared at a step count
    the others never reached."""
    a = _run(tmp_path, "a", [10_000_000, 20_000_000, 30_000_000])
    b = _run(tmp_path, "b", [10_000_000, 20_000_000])
    assert milestones([a, b]) == ["final.pt", "step_010000000.pt", "step_020000000.pt"]


def test_final_is_included_only_when_all_runs_finished(tmp_path):
    a = _run(tmp_path, "a", [10_000_000])
    b = _run(tmp_path, "b", [10_000_000], final=False)
    assert milestones([a, b]) == ["step_010000000.pt"]


def test_no_common_milestone_is_empty_not_a_crash(tmp_path):
    a = _run(tmp_path, "a", [10_000_000], final=False)
    b = _run(tmp_path, "b", [20_000_000], final=False)
    assert milestones([a, b]) == []


def test_a_single_run_has_no_spread():
    from hockey.seedspread import main
    import sys
    argv = sys.argv
    sys.argv = ["seedspread", "runs/only-one"]
    try:
        with pytest.raises(SystemExit):
            main()
    finally:
        sys.argv = argv
