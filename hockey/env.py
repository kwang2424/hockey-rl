"""Vectorised 1v1 ice hockey environment.

Everything is batched over ``num_envs`` in numpy -- there is no per-env Python
loop in the hot path. This is the single most important performance decision
in the project: a game-engine-backed env gives you a few thousand steps per
second, this gives you tens of thousands on one core, and that is the
difference between "trained overnight" and "trained over a month".

Conventions
-----------
* Agent 0 is team A and attacks +x. Agent 1 is team B and attacks -x.
* Observations are canonicalised by a 180 degree rotation for team B, so a
  single policy plays both sides. A 180 degree rotation (not an x-mirror) is
  used deliberately: it preserves chirality, so a policy that learned to turn
  left does not have to unlearn it when it swaps ends.
* Rewards are exactly zero-sum: ``r[:, 1] == -r[:, 0]``.
"""

import numpy as np

from .config import Config, DEFAULT
from . import rink

OBS_DIM = 34
ACT_DIM = 3          # [forward, turn, shoot], each in [-1, 1]
N_AGENTS = 2

EPS = 1e-9

# Named slices into the observation vector. Anything that reads observations
# (the scripted bot, analysis scripts, tests) should index through this rather
# than hardcoding offsets.
OBS_SLICES = {
    "own_vel":       slice(0, 2),    # body frame, / max_speed
    "omega":         slice(2, 3),
    "own_pos":       slice(3, 5),    # team-canonical, normalised to [-1, 1]
    "heading_rel":   slice(5, 7),    # cos/sin of heading vs. attack direction
    "puck_rel":      slice(7, 9),    # body frame, / rink_length
    "puck_dist":     slice(9, 10),
    "puck_relvel":   slice(10, 12),  # body frame, / puck_max_speed
    "blade_to_puck": slice(12, 14),
    "possession":    slice(14, 17),  # [mine, theirs, loose]
    "opp_rel":       slice(17, 19),
    "opp_relvel":    slice(19, 21),
    "opp_heading":   slice(21, 23),
    "atk_goal_rel":  slice(23, 25),
    "atk_goal_dist": slice(25, 26),
    "dfd_goal_rel":  slice(26, 28),
    "dfd_goal_dist": slice(28, 29),
    "board_depth":   slice(29, 30),
    "board_vec":     slice(30, 32),
    "shot_cooldown": slice(32, 33),
    "time_left":     slice(33, 34),
}


def _rotate_into_body(vec, theta):
    """Rotate world vectors into each agent's body frame.

    ``vec`` is (..., 2) and ``theta`` is (...). Note that for *relative*
    quantities the team-canonicalising 180 degree rotation cancels against the
    heading offset of pi, so no explicit team flip is needed here.
    """
    c, s = np.cos(theta), np.sin(theta)
    x = vec[..., 0] * c + vec[..., 1] * s
    y = -vec[..., 0] * s + vec[..., 1] * c
    return np.stack([x, y], axis=-1)


class VecHockeyEnv:
    """A batch of independent 1v1 hockey games."""

    obs_dim = OBS_DIM
    act_dim = ACT_DIM
    n_agents = N_AGENTS

    def __init__(self, num_envs: int = 64, cfg: Config = DEFAULT, seed: int = 0):
        self.n = num_envs
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        z2 = lambda *shape: np.zeros(shape, dtype=np.float64)
        self.skater_pos = z2(num_envs, N_AGENTS, 2)
        self.skater_vel = z2(num_envs, N_AGENTS, 2)
        self.theta = z2(num_envs, N_AGENTS)
        self.omega = z2(num_envs, N_AGENTS)
        self.puck_pos = z2(num_envs, 2)
        self.puck_vel = z2(num_envs, 2)
        self.cooldown = z2(num_envs, N_AGENTS)
        self.possessor = np.full(num_envs, -1, dtype=np.int64)
        self.step_count = np.zeros(num_envs, dtype=np.int64)

        # rolling per-episode bookkeeping, surfaced through ``info``
        self.ep_shots = np.zeros((num_envs, N_AGENTS), dtype=np.int64)
        self.ep_possession = np.zeros((num_envs, N_AGENTS), dtype=np.float64)

        self._goal_centers = rink.goal_centers(cfg)   # row 0: +x net, 1: -x net
        self.reset()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self):
        self._reset_idx(np.arange(self.n))
        return self.observe()

    def _reset_idx(self, idx):
        """Reset a subset of envs.

        Half the resets are a centre faceoff; the other half are randomised
        scrimmage states. The mix matters: pure faceoffs give the policy a very
        narrow state distribution and it never learns what to do after a
        broken play.
        """
        if idx.size == 0:
            return
        cfg, rng = self.cfg, self.rng
        k = idx.size

        faceoff = rng.random(k) < 0.5

        # --- puck ---
        puck = rink.sample_inside(cfg, rng, k, margin=1.0)
        puck[faceoff] = 0.0
        puck[faceoff] += rng.normal(0.0, 0.2, (int(faceoff.sum()), 2))
        self.puck_pos[idx] = puck
        pv = rng.normal(0.0, 2.5, (k, 2))
        pv[faceoff] = 0.0
        self.puck_vel[idx] = pv

        # --- skaters: each starts on its own half, facing the puck ---
        for a in range(N_AGENTS):
            sign = 1.0 if a == 0 else -1.0     # own half is behind the attack dir
            pos = rink.sample_inside(cfg, rng, k, margin=cfg.skater_radius + 0.6)
            pos[:, 0] = -sign * np.abs(pos[:, 0])          # own (defensive) half
            fo = np.zeros((k, 2))
            fo[:, 0] = -sign * (4.0 + rng.uniform(0, cfg.faceoff_jitter_pos, k))
            fo[:, 1] = rng.uniform(-3.0, 3.0, k)
            pos = np.where(faceoff[:, None], fo, pos)
            self.skater_pos[idx, a] = pos

            to_puck = puck - pos
            face = np.arctan2(to_puck[:, 1], to_puck[:, 0])
            self.theta[idx, a] = face + rng.normal(0.0, cfg.faceoff_jitter_heading, k)
            sv = rng.normal(0.0, 1.5, (k, 2))
            sv[faceoff] = 0.0
            self.skater_vel[idx, a] = sv
            self.omega[idx, a] = 0.0

        self.cooldown[idx] = 0.0
        self.possessor[idx] = -1
        self.step_count[idx] = 0
        self.ep_shots[idx] = 0
        self.ep_possession[idx] = 0.0

        # Separate any overlapping bodies produced by rejection sampling.
        for _ in range(4):
            self._resolve_skater_pairs()

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action):
        """Advance one control step. ``action`` is (n, 2, 3) in [-1, 1]."""
        cfg = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        phi_before = self._potential()

        self._apply_shots(action[..., 2])
        for _ in range(cfg.substeps):
            self._substep(action)

        self.step_count += 1
        scored_a, scored_b = rink.in_goal(cfg, self.puck_pos)
        goal = scored_a | scored_b
        truncated = (self.step_count >= cfg.max_episode_steps) & ~goal

        # Potential-based shaping, Ng/Harada/Russell 1999:
        #   F = gamma * Phi(s') - Phi(s)
        # This is provably policy-invariant, which is what stops the classic
        # "rock the puck back and forth to farm the shaping bonus" exploit that
        # a naive per-step progress bonus produces. Phi is defined to be 0 at
        # an absorbing state, hence the mask on goals.
        phi_after = np.where(goal, 0.0, self._potential())
        shaping = cfg.gamma * phi_after - phi_before

        r_a = cfg.goal_reward * (scored_a.astype(np.float64) - scored_b.astype(np.float64))
        r_a = r_a + shaping
        reward = np.stack([r_a, -r_a], axis=1)

        info = self._info(scored_a, scored_b, goal, truncated)

        done = goal | truncated
        if np.any(done):
            # The observation *before* the reset is the real s' for the
            # transition that just ended. Bootstrapping a time-limit
            # truncation off the post-reset observation would tell the policy
            # the clock running out is a state transition it caused.
            info["final_obs"] = self.observe()
            self._reset_idx(np.nonzero(done)[0])

        return self.observe(), reward, goal, truncated, info

    def _info(self, scored_a, scored_b, goal, truncated):
        return {
            "goal_a": scored_a.copy(),
            "goal_b": scored_b.copy(),
            "episode_end": (goal | truncated),
            "ep_len": self.step_count.copy(),
            "ep_shots": self.ep_shots.copy(),
            "ep_possession": self.ep_possession.copy(),
        }

    # ------------------------------------------------------------------
    # physics
    # ------------------------------------------------------------------
    def _apply_shots(self, shoot):
        """Release the puck for whichever possessor pulled the trigger."""
        cfg = self.cfg
        has = self.possessor >= 0
        if not np.any(has):
            return
        owner = np.clip(self.possessor, 0, N_AGENTS - 1)
        power = np.take_along_axis(shoot, owner[:, None], 1)[:, 0]
        fire = has & (power > cfg.shot_threshold)
        if not np.any(fire):
            return

        th = np.take_along_axis(self.theta, owner[:, None], 1)[:, 0]
        speed = cfg.shot_speed_min + (cfg.shot_speed_max - cfg.shot_speed_min) * np.clip(power, 0.0, 1.0)
        shot_v = np.stack([np.cos(th), np.sin(th)], axis=-1) * speed[:, None]

        self.puck_vel = np.where(fire[:, None], shot_v, self.puck_vel)
        np.add.at(self.ep_shots, (np.nonzero(fire)[0], owner[fire]), 1)
        self.cooldown[np.nonzero(fire)[0], owner[fire]] = cfg.shot_cooldown
        self.possessor = np.where(fire, -1, self.possessor)

    def _substep(self, action):
        cfg = self.cfg
        dt = cfg.physics_dt

        # Constraint order matters: body-vs-body contacts are resolved first
        # and the boards last, so the boards always get the final say. The
        # other order lets a puck pinned between a skater and the boards get
        # squeezed out of the rink. Overlapping bodies are a cosmetic
        # violation for one substep; leaving the rink is not.
        self._integrate_skaters(action, dt)
        self._resolve_skater_pairs()
        rink.resolve_goal_mouths(cfg, self.skater_pos, self.skater_vel,
                                 cfg.skater_radius, cfg.skater_restitution)
        rink.resolve_boards(cfg, self.skater_pos, self.skater_vel,
                            cfg.skater_radius, cfg.skater_restitution, 0.92)

        self.cooldown = np.maximum(self.cooldown - dt, 0.0)
        self._update_possession()
        self._integrate_puck(dt)

        rink.resolve_posts(cfg, self.puck_pos, self.puck_vel,
                           cfg.puck_radius, cfg.post_restitution)
        self._resolve_puck_bodies()
        rink.resolve_boards(cfg, self.puck_pos, self.puck_vel, cfg.puck_radius,
                            cfg.board_restitution, cfg.board_tangent_keep)

        owner = self.possessor
        for a in range(N_AGENTS):
            self.ep_possession[:, a] += np.where(owner == a, dt, 0.0)

    def _integrate_skaters(self, action, dt):
        """Thrust along the heading, then anisotropic (blade) friction.

        The lateral friction is capped at a finite grip budget, so turning hard
        at speed makes the skater slide instead of carving. That single line is
        most of what makes the motion read as skating rather than driving.
        """
        cfg = self.cfg
        fwd_cmd, turn_cmd = action[..., 0], action[..., 1]

        heading = np.stack([np.cos(self.theta), np.sin(self.theta)], axis=-1)
        accel_mag = np.where(fwd_cmd >= 0.0,
                             fwd_cmd * cfg.thrust_accel,
                             fwd_cmd * cfg.thrust_accel_back)
        accel = heading * accel_mag[..., None]

        v_fwd_s = np.sum(self.skater_vel * heading, axis=-1)
        v_fwd = heading * v_fwd_s[..., None]
        v_lat = self.skater_vel - v_fwd
        lat_speed = np.linalg.norm(v_lat, axis=-1)
        lat_dir = v_lat / np.maximum(lat_speed, EPS)[..., None]
        lat_decel = np.minimum(cfg.lateral_damp * lat_speed, cfg.grip_accel_max)

        accel = accel - lat_dir * lat_decel[..., None] - v_fwd * cfg.glide_damp

        self.skater_vel += accel * dt
        speed = np.linalg.norm(self.skater_vel, axis=-1, keepdims=True)
        self.skater_vel *= np.minimum(1.0, cfg.max_speed / np.maximum(speed, EPS))
        self.skater_pos += self.skater_vel * dt

        self.omega += (cfg.turn_accel * turn_cmd - cfg.ang_damp * self.omega) * dt
        np.clip(self.omega, -cfg.max_omega, cfg.max_omega, out=self.omega)
        self.theta = np.mod(self.theta + self.omega * dt + np.pi, 2 * np.pi) - np.pi

    def blade_points(self):
        """Where each skater's stick blade is, and how fast it is moving."""
        cfg = self.cfg
        heading = np.stack([np.cos(self.theta), np.sin(self.theta)], axis=-1)
        lateral = np.stack([-np.sin(self.theta), np.cos(self.theta)], axis=-1)
        pos = self.skater_pos + heading * cfg.blade_offset
        vel = self.skater_vel + lateral * (self.omega[..., None] * cfg.blade_offset)
        return pos, vel

    def _update_possession(self):
        """Whoever has a blade on the puck carries it; ties go to the closest.

        A current possessor keeps the puck while still in range, so possession
        does not flicker between two players in a battle.
        """
        cfg = self.cfg
        blade_pos, _ = self.blade_points()
        d = np.linalg.norm(self.puck_pos[:, None, :] - blade_pos, axis=-1)
        eligible = (d < cfg.capture_radius) & (self.cooldown <= 0.0)

        cur = np.clip(self.possessor, 0, N_AGENTS - 1)
        keeps = (self.possessor >= 0) & np.take_along_axis(eligible, cur[:, None], 1)[:, 0]

        masked = np.where(eligible, d, np.inf)
        best = np.argmin(masked, axis=1)
        any_elig = np.isfinite(masked.min(axis=1))
        fresh = np.where(any_elig, best, -1)

        self.possessor = np.where(keeps, self.possessor, fresh)

    def _integrate_puck(self, dt):
        """Carried pucks track the blade; loose pucks glide."""
        cfg = self.cfg
        blade_pos, blade_vel = self.blade_points()
        owner = np.clip(self.possessor, 0, N_AGENTS - 1)
        held = self.possessor >= 0

        bp = np.take_along_axis(blade_pos, owner[:, None, None], 1)[:, 0]
        bv = np.take_along_axis(blade_vel, owner[:, None, None], 1)[:, 0]
        correction = (bp - self.puck_pos) * cfg.carry_gain
        cmag = np.linalg.norm(correction, axis=-1, keepdims=True)
        correction *= np.minimum(1.0, cfg.carry_max_correction / np.maximum(cmag, EPS))
        carry_vel = bv + correction

        free_vel = self.puck_vel * np.exp(-cfg.puck_damp * dt)
        self.puck_vel = np.where(held[:, None], carry_vel, free_vel)

        speed = np.linalg.norm(self.puck_vel, axis=-1, keepdims=True)
        self.puck_vel *= np.minimum(1.0, cfg.puck_max_speed / np.maximum(speed, EPS))
        self.puck_pos += self.puck_vel * dt

    def _resolve_skater_pairs(self):
        """Equal-mass circle collision between the two skaters."""
        cfg = self.cfg
        d = self.skater_pos[:, 0] - self.skater_pos[:, 1]
        dist = np.linalg.norm(d, axis=-1)
        min_d = 2.0 * cfg.skater_radius
        hit = dist < min_d
        if not np.any(hit):
            return
        n = d / np.maximum(dist, EPS)[:, None]
        push = np.where(hit, (min_d - dist) * 0.5, 0.0)[:, None] * n
        self.skater_pos[:, 0] += push
        self.skater_pos[:, 1] -= push

        rel = np.sum((self.skater_vel[:, 0] - self.skater_vel[:, 1]) * n, axis=-1)
        approaching = hit & (rel < 0.0)
        j = np.where(approaching, -(1.0 + cfg.skater_restitution) * rel * 0.5, 0.0)[:, None] * n
        self.skater_vel[:, 0] += j
        self.skater_vel[:, 1] -= j

    def _resolve_puck_bodies(self):
        """Puck bouncing off skater bodies.

        Applied after the carry update, so a body check can knock a carried
        puck loose -- which is where puck battles along the boards come from.
        """
        cfg = self.cfg
        min_d = cfg.puck_radius + cfg.skater_radius
        for a in range(N_AGENTS):
            d = self.puck_pos - self.skater_pos[:, a]
            dist = np.linalg.norm(d, axis=-1)
            hit = (dist < min_d) & (self.possessor != a)
            if not np.any(hit):
                continue
            n = d / np.maximum(dist, EPS)[:, None]
            self.puck_pos += np.where(hit, min_d - dist, 0.0)[:, None] * n
            rel = np.sum((self.puck_vel - self.skater_vel[:, a]) * n, axis=-1)
            approaching = hit & (rel < 0.0)
            impulse = np.where(approaching, -(1.0 + cfg.puck_body_restitution) * rel, 0.0)
            self.puck_vel += impulse[:, None] * n
            # A hit knocks the puck off the stick.
            self.possessor = np.where(hit & (self.possessor >= 0), -1, self.possessor)

    # ------------------------------------------------------------------
    # reward
    # ------------------------------------------------------------------
    def _potential(self):
        """Phi(s) for team A. Team B's potential is exactly its negation."""
        cfg = self.cfg
        d_opp = np.linalg.norm(self.puck_pos - self._goal_centers[0], axis=-1)
        d_own = np.linalg.norm(self.puck_pos - self._goal_centers[1], axis=-1)
        phi = cfg.shaping_weight * (d_own - d_opp) / cfg.rink_length

        poss = np.where(self.possessor == 0, 1.0, np.where(self.possessor == 1, -1.0, 0.0))
        phi = phi + cfg.possession_weight * poss

        # Race-to-the-puck term. Antisymmetric by construction, so it cannot
        # break the zero-sum property, and potential-based, so it cannot
        # change which policy is optimal -- it only makes the first million
        # steps learnable.
        to_puck = np.linalg.norm(self.puck_pos[:, None, :] - self.skater_pos, axis=-1)
        phi = phi + cfg.proximity_weight * (to_puck[:, 1] - to_puck[:, 0]) / cfg.rink_length
        return phi

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------
    def observe(self):
        """Egocentric, team-canonical observations of shape (n, 2, OBS_DIM)."""
        cfg = self.cfg
        th = self.theta                                   # (n, 2)
        sign = np.array([1.0, -1.0])[None, :]             # team canonicalisation

        blade_pos, _ = self.blade_points()
        puck_p = self.puck_pos[:, None, :]
        puck_v = self.puck_vel[:, None, :]

        own_vel_b = _rotate_into_body(self.skater_vel, th) / cfg.max_speed
        omega = (self.omega / cfg.max_omega)[..., None]

        own_pos = self.skater_pos * sign[..., None]
        own_pos_n = own_pos / np.array([cfg.half_length, cfg.half_width])
        head_rel = np.stack([np.cos(th) * sign, np.sin(th) * sign], axis=-1)

        rel_puck = puck_p - self.skater_pos
        d_puck = np.linalg.norm(rel_puck, axis=-1, keepdims=True)
        rel_puck_b = _rotate_into_body(rel_puck, th) / cfg.rink_length
        rel_pv_b = _rotate_into_body(puck_v - self.skater_vel, th) / cfg.puck_max_speed
        blade_to_puck = _rotate_into_body(puck_p - blade_pos, th) / (4.0 * cfg.capture_radius)

        poss = np.zeros((self.n, N_AGENTS, 3))
        for a in range(N_AGENTS):
            poss[:, a, 0] = self.possessor == a               # I have it
            poss[:, a, 1] = (self.possessor >= 0) & (self.possessor != a)
            poss[:, a, 2] = self.possessor < 0                # loose

        other = self.skater_pos[:, ::-1, :]
        other_v = self.skater_vel[:, ::-1, :]
        other_th = th[:, ::-1]
        rel_opp_b = _rotate_into_body(other - self.skater_pos, th) / cfg.rink_length
        rel_oppv_b = _rotate_into_body(other_v - self.skater_vel, th) / cfg.max_speed
        dth = other_th - th
        opp_head = np.stack([np.cos(dth), np.sin(dth)], axis=-1)

        # Attack net: agent 0 shoots at the +x net (row 0), agent 1 at -x.
        atk = np.stack([self._goal_centers[0], self._goal_centers[1]])[None, :, :]
        dfd = np.stack([self._goal_centers[1], self._goal_centers[0]])[None, :, :]
        to_atk = atk - self.skater_pos
        to_dfd = dfd - self.skater_pos
        d_atk = np.linalg.norm(to_atk, axis=-1, keepdims=True)
        d_dfd = np.linalg.norm(to_dfd, axis=-1, keepdims=True)
        to_atk_b = _rotate_into_body(to_atk, th) / cfg.rink_length
        to_dfd_b = _rotate_into_body(to_dfd, th) / cfg.rink_length

        board_vec = rink.boundary_vector(cfg, self.skater_pos)
        board_depth = ((cfg.corner_radius - np.linalg.norm(board_vec, axis=-1))
                       / cfg.corner_radius)[..., None]
        board_n_b = _rotate_into_body(board_vec, th) / cfg.corner_radius

        cd = (self.cooldown / max(cfg.shot_cooldown, EPS))[..., None]
        t_left = np.broadcast_to(
            (1.0 - self.step_count / cfg.max_episode_steps)[:, None, None],
            (self.n, N_AGENTS, 1),
        )

        obs = np.concatenate(
            [
                own_vel_b, omega, own_pos_n, head_rel,
                rel_puck_b, d_puck / cfg.rink_length, rel_pv_b, blade_to_puck,
                poss,
                rel_opp_b, rel_oppv_b, opp_head,
                to_atk_b, d_atk / cfg.rink_length,
                to_dfd_b, d_dfd / cfg.rink_length,
                board_depth, board_n_b,
                cd, t_left,
            ],
            axis=-1,
        )
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    # ------------------------------------------------------------------
    def state_snapshot(self):
        """Everything the renderer needs for one frame."""
        return {
            "skater_pos": self.skater_pos.copy(),
            "theta": self.theta.copy(),
            "puck_pos": self.puck_pos.copy(),
            "possessor": self.possessor.copy(),
            "blade_pos": self.blade_points()[0],
        }
