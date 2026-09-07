"""Rink geometry: the SDF must agree with what the collision code believes."""

import numpy as np
import pytest

from hockey.config import DEFAULT as C
from hockey import rink


def test_sdf_sign_at_known_points():
    assert rink.sdf(C, np.array([0.0, 0.0])) < 0                    # centre ice
    assert rink.sdf(C, np.array([C.half_length - 0.05, 0.0])) < 0    # just inside end board
    assert rink.sdf(C, np.array([C.half_length + 1.0, 0.0])) > 0     # outside
    # The corner is cut: a point inside the bounding box but outside the
    # rounded rect must read as outside.
    assert rink.sdf(C, np.array([C.half_length - 0.5, C.half_width - 0.5])) > 0


def test_sdf_is_a_true_distance():
    """|sdf(a) - sdf(b)| <= |a - b| -- the 1-Lipschitz property of a real SDF."""
    rng = np.random.default_rng(0)
    a = rng.uniform(-40, 40, (2000, 2))
    b = a + rng.normal(0, 2.0, a.shape)
    lhs = np.abs(rink.sdf(C, a) - rink.sdf(C, b))
    rhs = np.linalg.norm(a - b, axis=-1)
    assert np.all(lhs <= rhs + 1e-9)


def test_normal_points_outward_and_is_unit():
    rng = np.random.default_rng(1)
    p = rng.uniform(-45, 45, (3000, 2))
    d, n = rink.sdf_and_normal(C, p)
    near = np.abs(d) < 6.0                       # where the normal is defined
    assert np.allclose(np.linalg.norm(n[near], axis=-1), 1.0, atol=1e-9)
    # Stepping along the normal must increase the signed distance.
    assert np.all(rink.sdf(C, p[near] + n[near] * 0.01) > d[near] - 1e-9)


def test_boundary_vector_is_rotation_equivariant():
    """The 180 degree team symmetry rests on this."""
    rng = np.random.default_rng(2)
    p = rink.sample_inside(C, rng, 2000, margin=0.0)
    assert np.allclose(rink.boundary_vector(C, p), -rink.boundary_vector(C, -p))


def test_sample_inside_respects_margin():
    rng = np.random.default_rng(3)
    for margin in (0.0, 1.0, 3.0):
        p = rink.sample_inside(C, rng, 500, margin=margin)
        assert np.all(rink.sdf(C, p) < -margin + 1e-9)


def test_resolve_boards_pushes_out_and_reflects():
    pos = np.array([[C.half_length + 2.0, 0.0]])
    vel = np.array([[8.0, 0.0]])                 # driving into the end board
    rink.resolve_boards(C, pos, vel, C.puck_radius, 0.5, 1.0)
    assert rink.sdf(C, pos)[0] + C.puck_radius <= 1e-9
    assert vel[0, 0] == pytest.approx(-4.0)      # reflected, restitution 0.5


def test_goal_detection():
    inside = np.array([[C.goal_line_x + 0.1, 0.0]])
    wide = np.array([[C.goal_line_x + 0.1, C.goal_half_width + 0.2]])
    short = np.array([[C.goal_line_x - 0.1, 0.0]])
    assert rink.in_goal(C, inside)[0][0] and not rink.in_goal(C, inside)[1][0]
    assert not rink.in_goal(C, wide)[0][0]
    assert not rink.in_goal(C, short)[0][0]
    assert rink.in_goal(C, -inside)[1][0]        # other end, other team
