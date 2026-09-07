"""The trajectory schema is a contract with the Godot/Unity viewers.

If these drift, the viewer breaks silently and shows a plausible-looking but
wrong game, so the schema is pinned here rather than only documented.
"""

import json

import numpy as np
import pytest

from hockey.config import DEFAULT as C
from hockey.bots import ChaseBot, StandStillBot
from hockey.export import record, SCHEMA_VERSION

REQUIRED_RINK_KEYS = {
    "length", "width", "corner_radius", "goal_line_x", "goal_half_width",
    "goal_depth", "skater_radius", "puck_radius", "blade_offset",
}


@pytest.fixture(scope="module")
def traj():
    return record(ChaseBot(), ChaseBot(), steps=400, seed=0, names=("A", "B"))


def test_header_and_rink_block(traj):
    assert traj["format"] == "hockey-rl-trajectory"
    assert traj["version"] == SCHEMA_VERSION
    assert traj["fps"] == pytest.approx(1.0 / C.control_dt)
    assert REQUIRED_RINK_KEYS <= set(traj["rink"])
    # The viewer draws the rink from these numbers, so they must match the sim.
    assert traj["rink"]["length"] == C.rink_length
    assert traj["rink"]["corner_radius"] == C.corner_radius
    assert traj["rink"]["goal_half_width"] == C.goal_half_width


def test_teams_attack_opposite_ends(traj):
    a, b = traj["teams"]
    assert a["attacks_x"] == 1 and b["attacks_x"] == -1
    assert a["colour"].startswith("#") and b["colour"].startswith("#")


def test_frames_are_absolute_state_and_in_bounds(traj):
    """Absolute, not deltas -- a viewer must be able to scrub to any index."""
    frames = traj["frames"]
    assert len(frames) == 400
    for i, f in enumerate(frames):
        assert set(f) >= {"t", "skaters", "puck", "possessor", "score"}
        assert f["t"] == pytest.approx(i * C.control_dt, abs=1e-3)
        assert len(f["skaters"]) == 2 and all(len(s) == 3 for s in f["skaters"])
        assert len(f["puck"]) == 2
        assert f["possessor"] in (-1, 0, 1)
        for s in f["skaters"]:
            assert abs(s[0]) <= C.half_length + 1.0
            assert abs(s[1]) <= C.half_width + 1.0
            assert -np.pi - 1e-6 <= s[2] <= np.pi + 1e-6


def test_score_is_monotonic_and_matches_events(traj):
    goals = [0, 0]
    for f in traj["frames"]:
        assert f["score"] == goals, "score must reflect goals scored so far"
        ev = f.get("event")
        if ev == "goal_a":
            goals[0] += 1
        elif ev == "goal_b":
            goals[1] += 1
    assert traj["final_score"] == goals


def test_events_are_from_the_known_set(traj):
    allowed = {"goal_a", "goal_b", "period_end"}
    seen = {f["event"] for f in traj["frames"] if "event" in f}
    assert seen <= allowed, f"unexpected event(s): {seen - allowed}"


def test_round_trips_through_json(traj):
    """Must contain no numpy scalars -- json.dump would raise on them."""
    text = json.dumps(traj)
    back = json.loads(text)
    assert back["frames"][10] == traj["frames"][10]


def test_possessor_matches_a_skater_near_the_puck(traj):
    """A possessed puck must actually be at that skater's blade."""
    reach = C.blade_offset + C.capture_radius + C.skater_radius
    for f in traj["frames"]:
        p = f["possessor"]
        if p < 0:
            continue
        sx, sy, _ = f["skaters"][p]
        d = np.hypot(f["puck"][0] - sx, f["puck"][1] - sy)
        assert d <= reach, f"possessor {p} is {d:.2f} m from the puck"


def test_a_still_game_still_exports_cleanly():
    t = record(StandStillBot(), StandStillBot(), steps=30, seed=1)
    assert len(t["frames"]) == 30
    assert t["final_score"] == [0, 0]
    json.dumps(t)
