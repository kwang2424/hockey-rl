"""Physical constants and tunables for the 1v1 ice hockey sim.

All units are SI: metres, seconds, kilograms, radians. Values are chosen to be
roughly NHL-plausible rather than exact -- the goal is that the *feel* of
skating (momentum, carving, sliding out of a hard turn) falls out of the
physics rather than being scripted.
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Config:
    # ---- rink geometry -------------------------------------------------
    # NHL rink is 200ft x 85ft with 28ft corner radius.
    rink_length: float = 60.0          # along x
    rink_width: float = 26.0           # along y
    corner_radius: float = 8.5

    goal_line_x: float = 26.0          # distance from centre ice to goal line
    goal_half_width: float = 0.915     # 6ft goal mouth
    goal_depth: float = 1.12           # cosmetic, used by the renderer
    post_radius: float = 0.05

    # ---- bodies --------------------------------------------------------
    puck_radius: float = 0.038
    skater_radius: float = 0.45

    # ---- integration ---------------------------------------------------
    physics_dt: float = 1.0 / 120.0
    substeps: int = 4                  # -> 30 Hz control rate
    max_episode_steps: int = 600       # 20 s at 30 Hz

    # ---- skating -------------------------------------------------------
    # Thrust is applied along the heading only; you cannot accelerate
    # sideways. Backward skating ("C-cuts") is deliberately weaker.
    thrust_accel: float = 7.0
    thrust_accel_back: float = 3.5
    max_speed: float = 11.0

    # Anisotropic friction: this is the whole trick. Lateral velocity is
    # killed hard (the blade edge bites) but only up to a finite grip
    # budget, so a hard turn at speed makes you drift instead of carving.
    lateral_damp: float = 20.0         # 1/s, applied to lateral velocity
    grip_accel_max: float = 14.0       # m/s^2 ceiling on that damping
    glide_damp: float = 0.35           # 1/s, forward direction (near-free glide)

    # ---- turning -------------------------------------------------------
    # These three interact: the steady-state turn rate under full command is
    # turn_accel / ang_damp, and max_omega only ever binds if that quotient
    # exceeds it. It previously did not -- turn_accel 16.0 with ang_damp 5.5
    # settles at 2.91 rad/s, so the declared 4.5 cap was unreachable dead
    # config and skaters pivoted 35% slower than the config claimed.
    # tests/test_physics.py now asserts the cap is actually reachable.
    turn_accel: float = 28.0           # rad/s^2 -> settles above max_omega
    ang_damp: float = 5.5              # 1/s
    max_omega: float = 4.5             # rad/s, ~0.70 s for a 180 pivot

    # ---- puck ----------------------------------------------------------
    puck_damp: float = 0.25            # 1/s on ice
    puck_max_speed: float = 40.0
    board_restitution: float = 0.55
    board_tangent_keep: float = 0.85
    post_restitution: float = 0.55

    # ---- stick / possession -------------------------------------------
    # v0 has no articulated stick. Instead each skater has a "blade point"
    # a fixed distance ahead of its centre; a puck inside the capture
    # radius of that point is carried. Reaching the blade into a loose puck
    # is how you pick it up, and how you poke it off an opponent.
    blade_offset: float = 0.85
    capture_radius: float = 0.42
    carry_gain: float = 18.0           # how hard the puck tracks the blade
    carry_max_correction: float = 9.0  # m/s, cap on that tracking velocity

    shot_threshold: float = 0.0        # action > this fires
    shot_speed_min: float = 14.0
    shot_speed_max: float = 30.0
    shot_cooldown: float = 0.25        # s, stops instant self-recapture

    # ---- contact -------------------------------------------------------
    skater_restitution: float = 0.30
    puck_body_restitution: float = 0.45

    # ---- reward --------------------------------------------------------
    goal_reward: float = 1.0
    # Potential-based shaping (Ng et al. 1999): F = gamma*Phi(s') - Phi(s).
    # Because it is potential-based it cannot create a reward loop -- the
    # optimal policy of the shaped MDP is the optimal policy of the real one.
    shaping_weight: float = 0.35
    possession_weight: float = 0.20

    # How much the puck-position term counts while the puck is LOOSE.
    #
    # Without this the position term pays for puck displacement regardless of
    # who ends up with it, which prices a 30 m/s clearance above an 11 m/s
    # controlled carry -- and the policy duly learned to fire on 99% of the
    # steps it held the puck, with 0% of shots on target. Discounting a loose
    # puck means advancing it *under control* is what pays.
    #
    # The factor is deliberately the same for both teams. A term that scaled
    # with *which* team holds the puck would be symmetric under the team swap
    # rather than antisymmetric, and would silently break the zero-sum
    # property; a shared factor times the antisymmetric position term stays
    # antisymmetric. tests/test_env.py enforces this.
    loose_puck_factor: float = 0.5

    # Per-step reward for holding the puck. THIS ONE IS NOT POTENTIAL-BASED,
    # and that is deliberate.
    #
    # A state potential cannot reward *duration*: it pays the transition into
    # possession and charges the transition out, so a 25-step possession nets
    # strictly less than a 1-step one (only the discount drag differs). No
    # value of possession_weight can make the policy want to carry the puck --
    # which is why raising it from 0.08 to 0.20 changed nothing.
    #
    # The cost of fixing that is the policy-invariance guarantee: unlike every
    # other term here, this one CAN change which policy is optimal, and its
    # failure mode is the classic one -- hoard the puck in a corner and never
    # shoot. hockey.diagnose flags exactly that (fire_while_holding near zero
    # with long possessions), so it is watched rather than assumed away.
    # Still exactly zero-sum: the holder gains it, the other skater loses it.
    #
    # The rate must exceed the discount drag while holding, or carrying still
    # loses money, and stay well under goal_reward/max_episode_steps or
    # hoarding for a whole episode competes with scoring. At 0.0008 a
    # full-episode hold is worth 0.48 against a goal's 1.00, so scoring
    # strictly dominates. The earlier 0.0015 made it 0.90 -- close enough to a
    # goal that the policy drifted towards keeping the puck and not shooting.
    # At gamma 0.995 no value satisfied both bounds at all.
    possession_rate: float = 0.0
    # Being closer to the puck than your opponent. Written as a *difference*
    # between the two skaters so the potential stays antisymmetric and the
    # reward stays exactly zero-sum. This is the term that gets a fresh policy
    # off the ground: possession and goals are both far too rare to bootstrap
    # from, but "skate at the puck harder than the other guy" has a gradient
    # on literally the first step.
    #
    # Weighted heavily on purpose. Under a random policy the puck-position
    # term contributes ~3.5x this term's reward variance, and almost none of
    # it is yet controllable -- so at a low weight the one signal a fresh
    # agent CAN act on is buried in noise, and measured learning goes
    # backwards. Because every term here is potential-based, reweighting
    # cannot change which policy is optimal; it only changes what is
    # learnable early.
    proximity_weight: float = 1.0
    # Measure proximity from the stick blade, not the body centre. Possession
    # requires the puck within capture_radius of the BLADE, which sits
    # blade_offset ahead of the skater -- so a body-centre potential rewards
    # closing on the puck without ever facing it, and the two objectives come
    # apart. Set False only to reproduce the old behaviour.
    proximity_from_blade: bool = True
    # Must match PPOConfig.gamma -- the shaping uses this one and GAE uses
    # that one, and if they diverge the shaping stops being policy-invariant.
    # tests/test_ppo.py asserts they agree.
    #
    # Back to 0.995, the incumbent's value. 0.998 was argued for on horizon
    # grounds (a 500-step horizon against a 600-step episode, versus 200) and
    # the argument still looks right -- but it shipped as part of v5, which
    # the ladder puts at 0.063, barely above random. Nothing here has been
    # tested in isolation, so the defaults track the best measured
    # configuration rather than the most persuasive reasoning.
    gamma: float = 0.995

    # ---- reset ---------------------------------------------------------
    faceoff_jitter_pos: float = 3.0
    faceoff_jitter_heading: float = 0.5

    # Curriculum: fraction of resets that start with the puck already on a
    # skater's blade, facing the net, with the defender goal-side. Scoring is
    # otherwise gated behind winning the puck first, so a fresh policy almost
    # never sees a goal and has nothing to attribute one to.
    #
    # Default 0.0 so evaluation always uses the honest faceoff distribution.
    # The trainer anneals its OWN env's `curriculum_puck_on_stick` attribute
    # from high to low; keeping the config default at zero is what stops a
    # curriculum-inflated number leaking into a reported score.
    puck_on_stick_prob: float = 0.0

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.substeps

    @property
    def half_length(self) -> float:
        return self.rink_length / 2.0

    @property
    def half_width(self) -> float:
        return self.rink_width / 2.0

    @property
    def inner_x(self) -> float:
        """Half-extent of the rounded rect's inner (sharp) rectangle."""
        return self.half_length - self.corner_radius

    @property
    def inner_y(self) -> float:
        return self.half_width - self.corner_radius

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = Config()
