"""Physics invariants. These are the tests that catch a sim quietly going wrong.

An RL agent will find and exploit any physics bug you leave in, and it will
look like "emergent behaviour" until you check. These pin down the things that
must be true no matter what the policy does.
"""

import numpy as np
import pytest

from hockey.config import DEFAULT as C
from hockey.env import VecHockeyEnv
from hockey import rink

BOARD_TOL = 2e-3   # first-order position correction on the curved corners


def _adversarial_rollout(steps=300, n=256, seed=0, forward=None):
    env = VecHockeyEnv(num_envs=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    worst_puck = worst_skater = 0.0
    for _ in range(steps):
        a = rng.uniform(-1, 1, (n, 2, 3))
        if forward is not None:
            a[..., 0] = forward
        env.step(a)
        worst_puck = max(worst_puck, float((rink.sdf(C, env.puck_pos) + C.puck_radius).max()))
        worst_skater = max(worst_skater, float((rink.sdf(C, env.skater_pos) + C.skater_radius).max()))
        assert np.isfinite(env.puck_pos).all() and np.isfinite(env.skater_pos).all()
    return env, worst_puck, worst_skater


def test_nothing_escapes_the_rink():
    _, wp, ws = _adversarial_rollout(forward=1.0)   # full throttle -> board pins
    assert wp < BOARD_TOL, f"puck escaped by {wp:.4f} m"
    assert ws < BOARD_TOL, f"skater escaped by {ws:.4f} m"


def test_no_tunnelling_at_max_shot_speed():
    """Fire the puck at the boards from every angle at max speed."""
    n = 360
    env = VecHockeyEnv(num_envs=n, seed=2)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    env.puck_pos[:] = 0.0
    env.puck_vel[:] = np.stack([np.cos(ang), np.sin(ang)], -1) * C.puck_max_speed
    env.possessor[:] = -1
    env.skater_pos[:] = np.array([[0.0, 12.0], [0.0, -12.0]])   # out of the way
    for _ in range(200):
        env.step(np.zeros((n, 2, 3)))
        assert (rink.sdf(C, env.puck_pos) + C.puck_radius).max() < BOARD_TOL


def test_speed_caps_hold():
    env, _, _ = _adversarial_rollout(steps=200, forward=1.0)
    assert np.linalg.norm(env.skater_vel, axis=-1).max() <= C.max_speed + 1e-6
    assert np.linalg.norm(env.puck_vel, axis=-1).max() <= C.puck_max_speed + 1e-6


def test_skating_is_anisotropic():
    """The whole point of the skating model: lateral velocity dies, forward glides.

    A skater given no input, moving sideways relative to its heading, must
    shed that lateral velocity far faster than a skater gliding forwards.
    """
    env = VecHockeyEnv(num_envs=2, seed=3)
    env.theta[:] = 0.0                       # facing +x
    env.omega[:] = 0.0
    env.skater_vel[:] = 0.0
    env.skater_vel[0, 0] = [6.0, 0.0]        # pure forward glide
    env.skater_vel[1, 0] = [0.0, 6.0]        # pure sideways drift
    env.puck_pos[:] = [0.0, -12.0]           # keep the puck out of it
    env.puck_vel[:] = 0.0
    for _ in range(15):
        env.step(np.zeros((2, 2, 3)))
    fwd = np.linalg.norm(env.skater_vel[0, 0])
    lat = np.linalg.norm(env.skater_vel[1, 0])
    # Forward glide is near-free: it should shed only the glide_damp amount
    # over this window (6 * exp(-glide_damp * 0.5) ~= 5.0).
    expected_glide = 6.0 * np.exp(-C.glide_damp * 15 * C.control_dt)
    assert fwd == pytest.approx(expected_glide, rel=0.05)
    # Lateral velocity is what the blade edge kills, and it must die far
    # faster than the forward glide. That ratio *is* the skating model.
    assert lat < 0.2 * fwd, f"lateral {lat:.2f} vs forward {fwd:.2f}: blade is not biting"


def test_hard_turn_at_speed_slides_out():
    """Grip is a finite budget, so a hard turn at speed should not be free."""
    env = VecHockeyEnv(num_envs=1, seed=4)
    env.theta[:] = 0.0
    env.omega[:] = 0.0
    env.skater_vel[:] = 0.0
    env.skater_vel[0, 0] = [C.max_speed, 0.0]
    env.puck_pos[:] = [0.0, -12.0]
    a = np.zeros((1, 2, 3))
    a[0, 0] = [0.0, 1.0, -1.0]               # crank the wheel, no thrust
    for _ in range(10):
        env.step(a)
    heading = np.array([np.cos(env.theta[0, 0]), np.sin(env.theta[0, 0])])
    v = env.skater_vel[0, 0]
    lateral = abs(float(v[0] * -heading[1] + v[1] * heading[0]))
    assert lateral > 0.5, "skater turned on rails; grip budget is not binding"


def test_skater_collision_conserves_momentum():
    env = VecHockeyEnv(num_envs=1, seed=5)
    env.skater_pos[0] = [[-0.6, 0.0], [0.6, 0.0]]
    env.skater_vel[0] = [[5.0, 0.0], [-5.0, 0.0]]
    env.theta[:] = 0.0
    env.omega[:] = 0.0
    env.puck_pos[:] = [0.0, -12.0]
    before = env.skater_vel[0].sum(axis=0).copy()
    env._resolve_skater_pairs()
    assert np.allclose(env.skater_vel[0].sum(axis=0), before, atol=1e-9)


def test_loose_puck_slows_down_and_never_speeds_up():
    env = VecHockeyEnv(num_envs=1, seed=6)
    env.puck_pos[:] = [0.0, 0.0]
    env.puck_vel[:] = [6.0, 0.0]
    env.possessor[:] = -1
    env.skater_pos[0] = [[-20.0, 10.0], [-20.0, -10.0]]   # far away
    prev = 6.0
    for _ in range(30):
        env.step(np.zeros((1, 2, 3)))
        env.possessor[:] = -1
        s = float(np.linalg.norm(env.puck_vel[0]))
        assert s <= prev + 1e-9
        prev = s
    assert prev < 6.0


def test_shot_leaves_the_stick_at_the_commanded_speed():
    env = VecHockeyEnv(num_envs=1, seed=7)
    env.theta[:] = 0.0
    env.omega[:] = 0.0
    env.skater_vel[:] = 0.0
    env.skater_pos[0] = [[0.0, 0.0], [-20.0, 10.0]]
    env.puck_pos[:] = [C.blade_offset, 0.0]               # sitting on the blade
    env.puck_vel[:] = 0.0
    env._update_possession()
    assert env.possessor[0] == 0, "puck on the blade should be possessed"
    env._apply_shots(np.array([[1.0, -1.0]]))
    assert env.possessor[0] == -1
    assert float(np.linalg.norm(env.puck_vel[0])) == pytest.approx(C.shot_speed_max, rel=1e-6)
    assert env.cooldown[0, 0] == pytest.approx(C.shot_cooldown)


def test_body_check_knocks_the_puck_loose():
    env = VecHockeyEnv(num_envs=1, seed=8)
    env.theta[:] = 0.0
    env.omega[:] = 0.0
    env.skater_vel[:] = 0.0
    env.skater_pos[0] = [[0.0, 0.0], [-20.0, 10.0]]
    env.puck_pos[:] = [C.blade_offset, 0.0]
    env.puck_vel[:] = 0.0
    env._update_possession()
    assert env.possessor[0] == 0
    # Drive the other skater straight through the puck.
    env.skater_pos[0, 1] = [C.blade_offset + 0.3, 0.0]
    env.skater_vel[0, 1] = [-8.0, 0.0]
    env._resolve_puck_bodies()
    assert env.possessor[0] == -1, "a body arriving on the puck should strip it"
