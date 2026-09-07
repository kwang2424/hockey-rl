"""Rink geometry as a signed distance field.

The rink is the Minkowski sum of a rectangle and a disc -- i.e. a rounded
rectangle. That makes the SDF exact and trivial:

    q = clamp(p, inner_rect)
    sdf(p) = |p - q| - corner_radius        (negative inside)
    outward normal = normalize(p - q)

which handles the straight boards and the rounded corners with the same three
lines, and gives the puck the correct "rims around the corner" behaviour for
free. Every function here is vectorised over a leading batch dimension.
"""

import numpy as np

from .config import Config

EPS = 1e-9


def sdf(cfg: Config, p: np.ndarray) -> np.ndarray:
    """Signed distance to the boards. Negative inside, positive outside."""
    q = _clamp_inner(cfg, p)
    return np.linalg.norm(p - q, axis=-1) - cfg.corner_radius


def boundary_vector(cfg: Config, p: np.ndarray) -> np.ndarray:
    """``p - clamp(p)``: points from the inner rectangle out towards the boards.

    Continuous everywhere (it decays smoothly to zero in the interior) and
    equivariant under the 180 degree team rotation, which the unit normal is
    not. Prefer this for observations.
    """
    return p - _clamp_inner(cfg, p)


def sdf_and_normal(cfg: Config, p: np.ndarray):
    """Signed distance plus the outward-pointing unit normal.

    Deep in the interior ``p == q`` and the normal is genuinely undefined. We
    return zero there rather than an arbitrary fixed direction: an arbitrary
    direction would break the 180 degree symmetry the whole team
    canonicalisation rests on. Contact requires |p - q| > corner_radius -
    body_radius > 0, so the degenerate branch can never fire while colliding.
    """
    d = boundary_vector(cfg, p)
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    unit = np.where(n > EPS, d / np.maximum(n, EPS), 0.0)
    return n[..., 0] - cfg.corner_radius, unit


def _clamp_inner(cfg: Config, p: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.clip(p[..., 0], -cfg.inner_x, cfg.inner_x),
            np.clip(p[..., 1], -cfg.inner_y, cfg.inner_y),
        ],
        axis=-1,
    )


def resolve_boards(cfg: Config, pos, vel, radius, restitution, tangent_keep):
    """Push bodies out of the boards and reflect their velocity, in place.

    Returns a boolean mask of which bodies were in contact this step.
    """
    d, n = sdf_and_normal(cfg, pos)
    pen = d + radius                      # >0 means overlapping the boards
    hit = pen > 0.0
    if not np.any(hit):
        return hit

    pos -= n * np.where(hit, pen, 0.0)[..., None]

    vn = np.sum(vel * n, axis=-1)         # normal component, +ve = outgoing
    outgoing = hit & (vn > 0.0)
    v_normal = n * vn[..., None]
    v_tangent = vel - v_normal
    new_vel = v_tangent * tangent_keep - v_normal * restitution
    vel[...] = np.where(outgoing[..., None], new_vel, vel)
    return hit


def goal_centers(cfg: Config) -> np.ndarray:
    """Centre of each goal mouth. Row 0 is the +x net, row 1 the -x net."""
    return np.array([[cfg.goal_line_x, 0.0], [-cfg.goal_line_x, 0.0]])


def post_positions(cfg: Config) -> np.ndarray:
    """The four goal posts as (4, 2)."""
    gx, gh = cfg.goal_line_x, cfg.goal_half_width
    return np.array([[gx, gh], [gx, -gh], [-gx, gh], [-gx, -gh]])


def resolve_posts(cfg: Config, pos, vel, radius, restitution):
    """Circle-vs-post collision for the puck, in place."""
    for post in post_positions(cfg):
        d = pos - post
        dist = np.linalg.norm(d, axis=-1)
        min_d = radius + cfg.post_radius
        hit = dist < min_d
        if not np.any(hit):
            continue
        n = d / np.maximum(dist, EPS)[..., None]
        pos += n * np.where(hit, min_d - dist, 0.0)[..., None]
        vn = np.sum(vel * n, axis=-1)
        approaching = hit & (vn < 0.0)
        vel -= np.where(approaching[..., None], (1.0 + restitution) * vn[..., None] * n, 0.0)


def resolve_goal_mouths(cfg: Config, pos, vel, radius, restitution):
    """Keep skaters out of the nets.

    The goal mouth is treated as a solid segment from post to post, so a
    skater can play *behind* the net (legal, and tactically interesting) but
    cannot stand inside it. The puck ignores this -- that is what scoring is.
    """
    gh = cfg.goal_half_width
    for gx in (cfg.goal_line_x, -cfg.goal_line_x):
        closest = np.stack(
            [np.full(pos.shape[:-1], gx), np.clip(pos[..., 1], -gh, gh)], axis=-1
        )
        d = pos - closest
        dist = np.linalg.norm(d, axis=-1)
        hit = dist < radius
        if not np.any(hit):
            continue
        n = d / np.maximum(dist, EPS)[..., None]
        pos += n * np.where(hit, radius - dist, 0.0)[..., None]
        vn = np.sum(vel * n, axis=-1)
        approaching = hit & (vn < 0.0)
        vel -= np.where(approaching[..., None], (1.0 + restitution) * vn[..., None] * n, 0.0)


def in_goal(cfg: Config, pos: np.ndarray):
    """(scored_plus_x, scored_minus_x) masks for a puck at ``pos``."""
    inside_mouth = np.abs(pos[..., 1]) <= cfg.goal_half_width
    return (
        inside_mouth & (pos[..., 0] > cfg.goal_line_x),
        inside_mouth & (pos[..., 0] < -cfg.goal_line_x),
    )


def sample_inside(cfg: Config, rng, n, margin=1.0):
    """Rejection-sample points comfortably inside the boards."""
    out = np.zeros((n, 2))
    todo = np.arange(n)
    while todo.size:
        cand = np.stack(
            [
                rng.uniform(-cfg.half_length, cfg.half_length, todo.size),
                rng.uniform(-cfg.half_width, cfg.half_width, todo.size),
            ],
            axis=-1,
        )
        ok = sdf(cfg, cand) < -margin
        out[todo[ok]] = cand[ok]
        todo = todo[~ok]
    return out
