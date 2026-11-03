"""Scripted and random policies.

These exist mainly so that "is it learning?" has an objective answer. A
learned policy that cannot beat ``ChaseBot`` is not learning, however good the
return curve looks.

Every policy here consumes observations rather than raw env state, which keeps
them drop-in interchangeable with a neural policy -- and doubles as a check
that the observation vector is actually sufficient to play the game.
"""

import numpy as np

from .config import Config, DEFAULT
from .env import OBS_SLICES


class Policy:
    """act(obs) -> action, where obs is (..., OBS_DIM) and action (..., 3)."""

    def act(self, obs, deterministic=True):
        raise NotImplementedError


class RandomBot(Policy):
    """Uniform noise. The floor: anything should beat this."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, deterministic=True):
        return self.rng.uniform(-1.0, 1.0, obs.shape[:-1] + (3,))


class StandStillBot(Policy):
    """Does nothing. Useful for isolating whether the sim itself is fair."""

    def act(self, obs, deterministic=True):
        return np.zeros(obs.shape[:-1] + (3,))


class ChaseBot(Policy):
    """A competent-but-dumb baseline: chase, carry to the net, shoot.

    No lookahead and no puck protection, but it does lead the puck when
    chasing and it does play goal-side when it has lost possession, which is
    enough to punish a policy that has only learned to skate at the puck.
    """

    def __init__(self, cfg: Config = DEFAULT, shoot_range=17.0, aim_tol=0.30,
                 lead_time=0.45, noise=0.05, seed=0):
        self.cfg = cfg
        self.shoot_range = shoot_range
        self.aim_tol = aim_tol
        self.lead_time = lead_time
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def act(self, obs, deterministic=True):
        cfg = self.cfg
        g = lambda k: obs[..., OBS_SLICES[k]]

        # Undo the observation normalisation to get metres and m/s back.
        puck_rel = g("puck_rel") * cfg.rink_length
        puck_relvel = g("puck_relvel") * cfg.puck_max_speed
        atk_rel = g("atk_goal_rel") * cfg.rink_length
        dfd_rel = g("dfd_goal_rel") * cfg.rink_length
        atk_dist = g("atk_goal_dist")[..., 0] * cfg.rink_length

        mine = g("possession")[..., 0] > 0.5
        theirs = g("possession")[..., 1] > 0.5

        # Carrying: drive at the net. Chasing: lead the puck. Defending:
        # sit between the puck and my own net rather than skating at the
        # carrier's back.
        intercept = puck_rel + puck_relvel * self.lead_time
        defend = 0.65 * puck_rel + 0.35 * dfd_rel
        target = np.where(mine[..., None], atk_rel,
                          np.where(theirs[..., None], defend, intercept))

        angle = np.arctan2(target[..., 1], target[..., 0])
        turn = np.clip(2.2 * angle, -1.0, 1.0)
        forward = np.clip(1.3 * np.cos(angle), -1.0, 1.0)

        net_angle = np.abs(np.arctan2(atk_rel[..., 1], atk_rel[..., 0]))
        shoot = np.where(
            mine & (atk_dist < self.shoot_range) & (net_angle < self.aim_tol), 1.0, -1.0
        )

        action = np.stack([forward, turn, shoot], axis=-1)
        if self.noise > 0 and not deterministic:
            action[..., :2] += self.rng.normal(0.0, self.noise, action[..., :2].shape)
        return np.clip(action, -1.0, 1.0)


class SniperBot(ChaseBot):
    """ChaseBot that carries in close and aims for a corner.

    ChaseBot aims at the centre of the net and fires anywhere inside 17 m
    once the net is within 0.30 rad of its heading. Against a goalie, which
    stands on exactly that line, it almost never scores, so it is no use as a
    reference for the goalie game. This bot carries the puck to within
    `shoot_range`, picks the corner farther from the goalie (when the
    observation includes the goalies; otherwise the corner nearer its
    current heading), and fires only once lined up on it within `aim_tol`.
    Off the puck it plays exactly like ChaseBot.
    """

    def __init__(self, cfg: Config = DEFAULT, shoot_range=9.0, aim_tol=0.03,
                 corner=0.75, **kw):
        super().__init__(cfg, shoot_range=shoot_range, aim_tol=aim_tol, **kw)
        self.corner = corner

    def act(self, obs, deterministic=True):
        from .env import GOALIE_OBS_SLICES
        cfg = self.cfg
        action = super().act(obs, deterministic)
        g = lambda k: obs[..., OBS_SLICES[k]]
        mine = g("possession")[..., 0] > 0.5
        atk_rel = g("atk_goal_rel") * cfg.rink_length
        h = g("heading_rel")                      # heading vs attack direction
        # The goal line's +y direction, in the body frame.
        lateral = np.stack([h[..., 1], h[..., 0]], axis=-1)
        corners = atk_rel[..., None, :] + np.array([1.0, -1.0])[:, None] * self.corner * lateral[..., None, :]
        if obs.shape[-1] >= GOALIE_OBS_SLICES["atk_goalie_rel"].stop:
            goalie = obs[..., GOALIE_OBS_SLICES["atk_goalie_rel"]] * cfg.rink_length
            far = np.linalg.norm(corners - goalie[..., None, :], axis=-1)
            pick = np.argmax(far, axis=-1)
        else:
            ang = np.abs(np.arctan2(corners[..., 1], corners[..., 0]))
            pick = np.argmin(ang, axis=-1)
        aim = np.take_along_axis(corners, pick[..., None, None], -2)[..., 0, :]
        dist = np.linalg.norm(aim, axis=-1)
        angle = np.arctan2(aim[..., 1], aim[..., 0])
        turn = np.clip(2.2 * angle, -1.0, 1.0)
        forward = np.clip(1.3 * np.cos(angle), -1.0, 1.0)
        shoot = np.where((dist < self.shoot_range) & (np.abs(angle) < self.aim_tol), 1.0, -1.0)
        carry = np.stack([forward, turn, shoot], axis=-1)
        if self.noise > 0 and not deterministic:
            carry[..., :2] += self.rng.normal(0.0, self.noise, carry[..., :2].shape)
        return np.where(mine[..., None], np.clip(carry, -1.0, 1.0), action)
