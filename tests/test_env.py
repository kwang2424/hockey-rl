"""Environment contract: shapes, determinism, termination, and reward structure."""

import numpy as np
import pytest

from hockey.config import DEFAULT as C, Config
from hockey.env import VecHockeyEnv, OBS_DIM, ACT_DIM, OBS_SLICES


def test_obs_slices_are_contiguous_and_complete():
    prev = 0
    for name, s in OBS_SLICES.items():
        assert s.start == prev, f"gap or overlap before {name}"
        prev = s.stop
    assert prev == OBS_DIM


def test_shapes_and_finiteness():
    env = VecHockeyEnv(num_envs=16, seed=0)
    obs = env.reset()
    assert obs.shape == (16, 2, OBS_DIM) and obs.dtype == np.float32
    obs, rew, goal, trunc, info = env.step(np.zeros((16, 2, ACT_DIM)))
    assert obs.shape == (16, 2, OBS_DIM)
    assert rew.shape == (16, 2)
    assert goal.shape == (16,) and trunc.shape == (16,)
    assert np.isfinite(obs).all() and np.isfinite(rew).all()


def test_actions_are_clipped_not_trusted():
    """Out-of-range actions must not be able to break the physics."""
    env = VecHockeyEnv(num_envs=8, seed=0)
    for _ in range(50):
        env.step(np.full((8, 2, ACT_DIM), 1e6))
    assert np.isfinite(env.skater_pos).all()
    assert np.linalg.norm(env.skater_vel, axis=-1).max() <= C.max_speed + 1e-6


def test_same_seed_same_trajectory():
    def roll(seed):
        env = VecHockeyEnv(num_envs=8, seed=seed)
        rng = np.random.default_rng(0)
        for _ in range(40):
            env.step(rng.uniform(-1, 1, (8, 2, ACT_DIM)))
        return env.puck_pos.copy()
    assert np.array_equal(roll(4), roll(4))
    assert not np.array_equal(roll(4), roll(5))


def test_reward_is_exactly_zero_sum():
    env = VecHockeyEnv(num_envs=64, seed=1)
    rng = np.random.default_rng(2)
    for _ in range(200):
        _, rew, _, _, _ = env.step(rng.uniform(-1, 1, (64, 2, ACT_DIM)))
        assert np.array_equal(rew[:, 0], -rew[:, 1])


def test_reward_decomposes_into_exactly_three_known_terms():
    """reward = goal + [gamma*Phi(s') - Phi(s)] + possession_rate.

    The middle bracket must be *exactly* a potential difference: that is what
    makes it policy-invariant and unfarmable, and a mismatch here means the
    shaping has grown a cycle. The third term is deliberately outside the
    potential (a potential cannot reward duration -- see
    Config.possession_rate), so it is pinned separately rather than folded in.
    """
    env = VecHockeyEnv(num_envs=64, seed=3)
    rng = np.random.default_rng(4)
    saw_possession = False
    for _ in range(150):
        phi_before = env._potential().copy()
        _, rew, goal, _, info = env.step(rng.uniform(-1, 1, (64, 2, ACT_DIM)))
        goal_term = C.goal_reward * (info["goal_a"].astype(float) - info["goal_b"].astype(float))
        # Phi is 0 at an absorbing state; on a goal the env resets immediately,
        # so reconstruct that case from the goal mask rather than the new state.
        ended = info["episode_end"]
        phi_after = np.where(goal, 0.0, env._potential())
        # `possessor` is post-step, and reset would clobber it, so the exact
        # check is restricted to envs that did not end this step.
        rate = C.possession_rate * np.where(env.possessor == 0, 1.0,
                                            np.where(env.possessor == 1, -1.0, 0.0))
        saw_possession |= bool((env.possessor[~ended] >= 0).any())
        expected = goal_term + C.gamma * phi_after - phi_before + rate
        assert np.allclose(rew[~ended, 0], expected[~ended], atol=1e-12)
    assert saw_possession, "test never exercised the possession-rate branch"


def test_possession_rate_is_zero_sum_and_pays_per_step():
    """Unlike the potential terms, this one must reward *duration*.

    The mechanism is tested with the rate switched on explicitly, because the
    default is 0.0: it shipped as part of v5, which the ladder puts barely
    above random, so it is available rather than assumed. Turn it on with
    --set possession_rate=<v>.
    """
    cfg_on = Config(possession_rate=0.0015)
    env = VecHockeyEnv(num_envs=1, cfg=cfg_on, seed=13)
    env.skater_pos[0] = [[0.0, 0.0], [-20.0, 9.0]]
    env.skater_vel[:] = 0.0
    env.theta[0] = [0.0, 0.0]
    env.omega[:] = 0.0
    env.puck_pos[:] = [cfg_on.blade_offset, 0.0]
    env.puck_vel[:] = 0.0
    env._update_possession()
    assert env.possessor[0] == 0

    total = 0.0
    hold_steps = 0
    for _ in range(20):
        act = np.zeros((1, 2, ACT_DIM))
        act[0, 0] = [0.0, 0.0, -1.0]          # hold: do not shoot
        _, rew, _, _, _ = env.step(act)
        assert rew[0, 0] == pytest.approx(-rew[0, 1])
        if env.possessor[0] == 0:
            hold_steps += 1
            total += float(rew[0, 0])
    assert hold_steps > 10, "skater failed to keep the puck on its blade"

    # Differential check: rerun the identical scenario with the rate switched
    # off. The gap must be exactly rate * held steps, which isolates this term
    # from the discount drag the potential charges for sitting in a good state.
    off = Config(possession_rate=0.0)
    env2 = VecHockeyEnv(num_envs=1, cfg=off, seed=13)
    env2.skater_pos[0] = [[0.0, 0.0], [-20.0, 9.0]]
    env2.skater_vel[:] = 0.0
    env2.theta[0] = [0.0, 0.0]
    env2.omega[:] = 0.0
    env2.puck_pos[:] = [off.blade_offset, 0.0]
    env2.puck_vel[:] = 0.0
    env2._update_possession()
    total_off, hold_off = 0.0, 0
    for _ in range(20):
        act = np.zeros((1, 2, ACT_DIM))
        act[0, 0] = [0.0, 0.0, -1.0]
        _, rew, _, _, _ = env2.step(act)
        if env2.possessor[0] == 0:
            hold_off += 1
            total_off += float(rew[0, 0])
    assert hold_off == hold_steps, "the two runs must be physically identical"
    assert total - total_off == pytest.approx(cfg_on.possession_rate * hold_steps, abs=1e-9)

    # The differential above is the real invariant. Whether holding ends up
    # net *positive* depends on gamma: the discount drag is (1-gamma)*Phi, and
    # at the incumbent's gamma of 0.995 the drag exceeds every rate that also
    # keeps a full-episode hoard below a goal. That window is empty here, and
    # the earlier version of this test quietly assumed it was not.
    assert total_off < 0.0, "a potential alone makes sitting on the puck cost you"
    assert total > total_off


def test_shaping_cannot_be_farmed_by_a_closed_loop():
    """Returning the puck to where it started must not yield free reward.

    The skaters are pinned completely -- position, velocity, heading and spin
    -- so that the puck really is the only thing that moves and the loop is a
    genuine closed loop *in state*. Leaving them free to drift and rotate
    makes the potential change for reasons that have nothing to do with the
    puck, which quietly weakens the test.
    """
    env = VecHockeyEnv(num_envs=1, seed=5)
    pinned_pos = np.array([[-20.0, 10.0], [-20.0, -10.0]])
    pinned_theta = np.array([0.3, -1.1])

    def pin():
        env.skater_pos[0] = pinned_pos
        env.skater_vel[0] = 0.0
        env.theta[0] = pinned_theta
        env.omega[0] = 0.0
        env.possessor[:] = -1

    pin()
    env.puck_pos[:] = [0.0, 0.0]
    phi_start = float(env._potential()[0])
    total, phi_sum = 0.0, 0.0
    for cycle in range(20):                      # push the puck out and back
        for direction in (1.0, -1.0):
            for _ in range(10):
                env.puck_vel[:] = [6.0 * direction, 0.0]
                _, rew, _, _, _ = env.step(np.zeros((1, 2, ACT_DIM)))
                pin()
                total += float(rew[0, 0])
                phi_sum += float(env._potential()[0])

    # The puck ends where it began and nothing else has moved, so the
    # potential must too.
    assert float(env._potential()[0]) == pytest.approx(phi_start, abs=1e-9)

    # The accumulated shaping is not quite zero, and the reason matters:
    #   sum[gamma*Phi(s') - Phi(s)] = (Phi_end - Phi_start) - (1-gamma)*sum Phi
    # The first bracket vanishes on a closed loop, leaving only the discount
    # drag -- which is NEGATIVE. So a closed loop strictly *costs* reward
    # rather than paying it, and dithering is penalised. That is the property
    # worth asserting; the magnitude is predicted, not merely bounded.
    assert total < 0.0, f"closed loop paid {total:+.6f} -- shaping is farmable"
    assert total == pytest.approx(-(1.0 - C.gamma) * phi_sum, rel=1e-6)


def test_closed_loop_pays_exactly_zero_when_undiscounted():
    """With gamma = 1 the telescoping is exact and the loop is free, not costly."""
    cfg = Config(gamma=1.0)
    env = VecHockeyEnv(num_envs=1, cfg=cfg, seed=5)
    pinned_pos = np.array([[-20.0, 10.0], [-20.0, -10.0]])
    pinned_theta = np.array([0.3, -1.1])

    def pin():
        env.skater_pos[0] = pinned_pos
        env.skater_vel[0] = 0.0
        env.theta[0] = pinned_theta
        env.omega[0] = 0.0
        env.possessor[:] = -1

    pin()
    env.puck_pos[:] = [0.0, 0.0]
    total = 0.0
    for cycle in range(20):
        for direction in (1.0, -1.0):
            for _ in range(10):
                env.puck_vel[:] = [6.0 * direction, 0.0]
                _, rew, _, _, _ = env.step(np.zeros((1, 2, ACT_DIM)))
                pin()
                total += float(rew[0, 0])
    assert total == pytest.approx(0.0, abs=1e-9), f"loop paid {total:+.3e}"


def test_episode_truncates_at_the_step_limit():
    cfg = Config(max_episode_steps=25)
    env = VecHockeyEnv(num_envs=4, cfg=cfg, seed=6)
    # Park everything far from both nets so no goal can end it early.
    for t in range(25):
        env.skater_pos[:] = np.array([[0.0, 10.0], [0.0, -10.0]])
        env.puck_pos[:] = [0.0, 0.0]
        env.puck_vel[:] = 0.0
        _, _, goal, trunc, _ = env.step(np.zeros((4, 2, ACT_DIM)))
    assert trunc.all() and not goal.any()
    assert (env.step_count == 0).all(), "envs should have reset after truncation"


def test_goal_ends_the_episode_and_scores_the_right_way():
    env = VecHockeyEnv(num_envs=2, seed=7)
    env.puck_pos[0] = [C.goal_line_x - 0.2, 0.0]
    env.puck_vel[0] = [20.0, 0.0]                # into the +x net: team A scores
    env.puck_pos[1] = [-C.goal_line_x + 0.2, 0.0]
    env.puck_vel[1] = [-20.0, 0.0]               # into the -x net: team B scores
    env.possessor[:] = -1
    phi_before = env._potential().copy()
    _, rew, goal, _, info = env.step(np.zeros((2, 2, ACT_DIM)))
    assert goal.all()
    assert info["goal_a"][0] and not info["goal_b"][0]
    assert info["goal_b"][1] and not info["goal_a"][1]
    # The goal itself pays goal_reward minus the potential already banked for
    # getting the puck down there -- not the full goal_reward. The shaping is
    # an advance on the goal, not a bonus on top of it; the episode total is
    # what telescopes back to goal_reward (see the test below).
    assert rew[0, 0] == pytest.approx(C.goal_reward - phi_before[0])
    assert rew[1, 0] == pytest.approx(-C.goal_reward - phi_before[1])
    assert rew[0, 0] > 0 and rew[1, 0] < 0


def test_shaped_return_is_path_independent():
    """Longer routes to the same goal must pay exactly the same total.

    With gamma = 1 the shaping terms telescope, so the undiscounted episode
    return is goal_reward - Phi(s_0): it depends only on where the puck
    started and whether it went in, never on the route it took. That is the
    operational statement of "potential-based shaping cannot change the
    optimal policy" -- no scenic route pays better, so there is nothing to
    farm. A naive per-step "puck moved towards the goal" bonus fails this.
    """
    cfg = Config(gamma=1.0)

    def run(wobble):
        env = VecHockeyEnv(num_envs=1, cfg=cfg, seed=11)
        env.skater_pos[0] = [[-20.0, 11.0], [-20.0, -11.0]]
        env.skater_vel[:] = 0.0
        env.puck_pos[:] = [-10.0, 0.0]
        env.puck_vel[:] = 0.0
        env.possessor[:] = -1
        phi0 = float(env._potential()[0])
        total, path_len = 0.0, 0.0
        prev = env.puck_pos[0].copy()
        for step in range(600):
            # One full period of lateral wobble, so every route ends up back
            # on the centre line and scores -- but covers a different distance
            # getting there.
            lateral = wobble * np.sin(2 * np.pi * step / 80.0) if step < 80 else 0.0
            env.puck_vel[:] = [6.0, lateral]
            _, rew, goal, _, _ = env.step(np.zeros((1, 2, ACT_DIM)))
            env.possessor[:] = -1
            total += float(rew[0, 0])
            path_len += float(np.linalg.norm(env.puck_pos[0] - prev))
            prev = env.puck_pos[0].copy()
            if goal[0]:
                return total, phi0, path_len
        raise AssertionError(f"puck never reached the net (wobble={wobble})")

    direct, phi0, len_direct = run(0.0)
    scenic, _, len_scenic = run(8.0)

    assert len_scenic > len_direct * 1.05, "the two routes should differ in length"
    assert direct == pytest.approx(cfg.goal_reward - phi0, abs=1e-12)
    assert scenic == pytest.approx(direct, abs=1e-12), (
        f"the {len_scenic:.0f}m route paid {scenic - direct:+.2e} more than "
        f"the {len_direct:.0f}m route -- the shaping has a farmable cycle"
    )


def test_final_obs_is_pre_reset_state():
    """Bootstrapping needs the real s', not the state after the faceoff."""
    env = VecHockeyEnv(num_envs=2, seed=8)
    env.puck_pos[0] = [C.goal_line_x - 0.2, 0.0]
    env.puck_vel[0] = [20.0, 0.0]
    obs, _, goal, _, info = env.step(np.zeros((2, 2, ACT_DIM)))
    assert goal[0] and "final_obs" in info
    puck_dist = OBS_SLICES["atk_goal_dist"]
    # In the pre-reset observation the puck is in the net; after the reset the
    # game has been re-centred, so the two must differ.
    assert not np.allclose(info["final_obs"][0], obs[0])


def test_possession_is_exclusive():
    env = VecHockeyEnv(num_envs=64, seed=9)
    rng = np.random.default_rng(10)
    for _ in range(300):
        env.step(rng.uniform(-1, 1, (64, 2, ACT_DIM)))
        assert set(np.unique(env.possessor)).issubset({-1, 0, 1})
        obs = env.observe()
        poss = obs[..., OBS_SLICES["possession"]]
        assert np.allclose(poss.sum(-1), 1.0), "possession one-hot must sum to 1"
        both = (poss[:, 0, 0] > 0.5) & (poss[:, 1, 0] > 0.5)
        assert not both.any(), "two players cannot both own the puck"


def test_loose_puck_gate_preserves_zero_sum_and_team_symmetry():
    """The control gate must be team-agnostic.

    A gate that scaled with *which* team holds the puck would be symmetric
    under the 180-degree team swap, and multiplying it into the antisymmetric
    position term would make the product symmetric -- quietly destroying the
    zero-sum property the whole self-play setup depends on. This pins that the
    gate is shared.
    """
    env = VecHockeyEnv(num_envs=256, seed=21)
    rng = np.random.default_rng(5)
    saw_loose = saw_held = False
    for _ in range(400):
        _, rew, _, _, _ = env.step(rng.uniform(-1, 1, (256, 2, ACT_DIM)))
        assert np.array_equal(rew[:, 0], -rew[:, 1])
        saw_loose |= bool((env.possessor < 0).any())
        saw_held |= bool((env.possessor >= 0).any())
    assert saw_loose and saw_held, "test never exercised both gate branches"

    # And the mirrored world must produce exactly mirrored potentials.
    m = VecHockeyEnv(num_envs=256, seed=21)
    m.skater_pos = -env.skater_pos[:, ::-1].copy()
    m.skater_vel = -env.skater_vel[:, ::-1].copy()
    m.theta = (env.theta[:, ::-1] + np.pi).copy()
    m.omega = env.omega[:, ::-1].copy()
    m.puck_pos = -env.puck_pos.copy()
    m.puck_vel = -env.puck_vel.copy()
    m.possessor = np.where(env.possessor < 0, -1, 1 - env.possessor)
    assert np.allclose(env._potential(), -m._potential(), atol=1e-12)


def test_carrying_the_puck_beats_flinging_it_away():
    """Advancing the puck under control must out-earn an uncontrolled clear.

    This is the property the previous reward got backwards, and it is what
    made the policy shoot on 99% of the steps it held the puck.
    """
    def advance(possessed):
        env = VecHockeyEnv(num_envs=1, seed=31)
        env.skater_pos[0] = [[-4.0, 0.0], [-20.0, 8.0]]
        env.skater_vel[:] = 0.0
        env.theta[0] = [0.0, 0.0]
        env.omega[:] = 0.0
        env.puck_pos[:] = [0.0, 0.0]
        env.possessor[:] = 0 if possessed else -1
        before = float(env._potential()[0])
        env.puck_pos[:] = [10.0, 0.0]          # same 10 m of progress either way
        return float(env._potential()[0]) - before

    controlled = advance(True)
    loose = advance(False)
    assert controlled > loose > 0, (
        f"controlled advance paid {controlled:+.4f}, loose paid {loose:+.4f}"
    )
    # The size of this margin is a deliberate trade-off, not a free parameter:
    # it *is* the potential a shot gives up, so a wider margin directly raises
    # the minimum shot success rate worth taking. At loose_puck_factor 0.5 the
    # margin was 2x and the bar was 30.5%, and the policy stopped shooting
    # altogether. Keep it clear but modest; test_taking_a_shot_is_not_priced_
    # out_of_reach pins the other end.
    assert controlled > 1.2 * loose, (
        f"margin only {controlled/loose:.2f}x -- controlled carry barely beats a clear"
    )
