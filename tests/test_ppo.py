"""Trainer-level regression tests."""

import numpy as np
import pytest
import torch

from rl.ppo import PPOTrainer, PPOConfig
from rl.nets import RunningNorm, ActorCritic
from hockey.env import OBS_DIM, ACT_DIM


def _tiny():
    return PPOTrainer(PPOConfig(num_envs=8, rollout_steps=8, num_minibatches=8, pool_size=2))


def test_learn_on_ragged_batch_stays_finite():
    """A ragged batch must never produce a single-sample minibatch.

    The learner mask (agents driven by a frozen pool opponent are excluded)
    makes the batch size arbitrary. A minibatch of one has an undefined
    unbiased std, and dividing advantages by it turns every weight into NaN.
    This is a regression test for exactly that.
    """
    t = _tiny()
    rng = np.random.default_rng(0)
    for n in (129, 257, 1001, 8 * 7 + 1):        # sizes that leave a remainder of 1
        b = {
            "obs": rng.normal(0, 1, (n, OBS_DIM)).astype(np.float32),
            "act": rng.uniform(-1, 1, (n, ACT_DIM)).astype(np.float32),
            "logp": rng.normal(0, 1, n).astype(np.float32),
            "adv": rng.normal(0, 1, n).astype(np.float32),
            "ret": rng.normal(0, 1, n).astype(np.float32),
            "val": rng.normal(0, 1, n).astype(np.float32),
            "mask": np.ones(n, dtype=np.float32),
        }
        t.learn_on(b)
        params = torch.cat([p.flatten() for p in t.net.parameters()])
        assert torch.isfinite(params).all(), f"non-finite params after n={n}"


def test_nonfinite_gradients_are_skipped_not_applied():
    t = _tiny()
    before = torch.cat([p.flatten() for p in t.net.parameters()]).clone()
    batch, _ = t.collect()
    batch["adv"][:] = np.inf                      # force a poisoned update
    info = t.learn_on(batch)
    after = torch.cat([p.flatten() for p in t.net.parameters()])
    assert torch.isfinite(after).all()
    assert info["nonfinite_skips"] > 0
    assert torch.allclose(before, after), "a poisoned update should be a no-op"


def test_collect_masks_pool_controlled_agents():
    t = _tiny()
    t._snapshot()
    t.opp_id[:] = 0                               # every team B is a frozen bot
    batch, stats = t.collect()
    mask = batch["mask"].reshape(-1, 2)
    assert mask[:, 0].all(), "team A is always the learner"
    assert not mask[:, 1].any(), "pool-driven team B must not be trained on"


def test_gae_bootstraps_through_truncation_but_not_goals():
    t = _tiny()
    T, N = 4, 2
    rew = np.zeros((T, N, 2), np.float32)
    val = np.ones((T, N, 2), np.float32) * 2.0
    tval = np.ones((T, N, 2), np.float32) * 2.0
    done = np.zeros((T, N), np.float32)
    trunc = np.zeros((T, N), np.float32)
    done[-1, 0] = 1.0                              # env 0 ends on a goal
    trunc[-1, 1] = 1.0                             # env 1 just runs out of clock
    adv, _ = t._gae(rew, val, done, trunc, np.ones((N, 2), np.float32) * 2.0, tval)
    # A goal is a real terminal: no bootstrap, so the whole value is a surprise.
    assert adv[-1, 0, 0] < -1.5
    # Truncation is not: bootstrapping off the true final state leaves ~no error.
    assert abs(adv[-1, 1, 0]) < 0.05


def test_running_norm_matches_batch_statistics():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, (5000, 4))
    rn = RunningNorm(4)
    for chunk in np.array_split(x, 25):
        rn.update(chunk)
    assert np.allclose(rn.mean, x.mean(0), atol=1e-6)
    assert np.allclose(rn.var, x.var(0), atol=1e-6)


def test_curriculum_anneals_and_never_leaks_into_eval():
    """The curriculum must apply to the training env only.

    If evaluation inherited it, every reported score would be measured on
    hand-delivered scoring chances and would look far better than the policy
    is.
    """
    from hockey.config import DEFAULT
    from hockey.env import VecHockeyEnv

    t = PPOTrainer(PPOConfig(num_envs=8, rollout_steps=8, curriculum_start=0.8,
                             curriculum_end=0.1, curriculum_frac=0.5))
    assert t.set_curriculum(0.0) == pytest.approx(0.8)
    assert t.set_curriculum(0.25) == pytest.approx(0.45)
    assert t.set_curriculum(0.5) == pytest.approx(0.1)
    assert t.set_curriculum(1.0) == pytest.approx(0.1), "must clamp, not overshoot"
    assert t.env.curriculum_puck_on_stick == pytest.approx(0.1)

    # A freshly built env -- which is what evaluation uses -- is unaffected.
    assert DEFAULT.puck_on_stick_prob == 0.0
    assert VecHockeyEnv(num_envs=4, seed=0).curriculum_puck_on_stick == 0.0


def test_curriculum_starts_are_real_scoring_chances_not_free_goals():
    """The defender must actually be between the carrier and the net."""
    import numpy as np
    from hockey.config import DEFAULT
    from hockey.env import VecHockeyEnv

    env = VecHockeyEnv(num_envs=1024, seed=3)
    env.curriculum_puck_on_stick = 1.0
    env._reset_idx(np.arange(1024))

    carrier = env.possessor
    assert (carrier >= 0).all()
    other = 1 - carrier
    rows = np.arange(1024)
    net = np.where(carrier[:, None] == 0,
                   np.array([DEFAULT.goal_line_x, 0.0]),
                   np.array([-DEFAULT.goal_line_x, 0.0]))
    cpos = env.skater_pos[rows, carrier]
    dpos = env.skater_pos[rows, other]
    # Defender is closer to the net than the carrier is, most of the time.
    closer = np.linalg.norm(dpos - net, axis=-1) < np.linalg.norm(cpos - net, axis=-1)
    assert closer.mean() > 0.8, f"defender goal-side only {closer.mean():.2f} of the time"


def test_policy_mean_stays_inside_the_action_range():
    """An unbounded mean can walk outside [-1, 1] and kill exploration.

    Once the mean sits far enough outside the clipped range every sample
    executes identically, so the environment cannot tell the policy apart from
    a constant -- and no reward change can pull it back. This is a regression
    test for the real failure: the shoot mean reached +3.48 (sigma 1.29), so
    roughly 1 possession in 278 sampled "do not shoot".
    """
    from hockey.env import OBS_DIM
    net = ActorCritic(OBS_DIM, ACT_DIM, bounded_mean=True)
    # Drive the pre-squash output hard in both directions.
    final = [m for m in net.actor if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        final.bias.fill_(50.0)
    obs = torch.randn(256, OBS_DIM)
    mean = net.dist(obs).mean
    assert torch.all(mean.abs() <= 1.0), f"mean escaped to {mean.abs().max():.2f}"

    with torch.no_grad():
        final.bias.fill_(-50.0)
    assert torch.all(net.dist(obs).mean.abs() <= 1.0)


def test_exploration_noise_has_a_ceiling():
    """sigma grew 0.607 -> 1.295 over training; the ceiling stops the runaway."""
    from hockey.env import OBS_DIM
    net = ActorCritic(OBS_DIM, ACT_DIM, bounded_mean=True, max_log_std=0.0)
    with torch.no_grad():
        net.log_std.fill_(5.0)
    sigma = net.dist(torch.randn(8, OBS_DIM)).scale
    assert torch.all(sigma <= 1.0 + 1e-6), f"sigma reached {sigma.max():.2f}"


def test_a_bounded_mean_keeps_both_sides_of_a_threshold_reachable():
    """With the mean saturated at its bound, refusing to fire must stay likely."""
    from hockey.env import OBS_DIM
    from math import erf, sqrt
    net = ActorCritic(OBS_DIM, ACT_DIM, init_log_std=-0.5, bounded_mean=True)
    sigma = float(net.log_std.exp()[2])
    p_not_fire = 0.5 * (1 - erf((1.0 / sigma) / sqrt(2)))   # worst case, mean = +1
    assert p_not_fire > 0.02, (
        f"even at the bound only {p_not_fire:.4f} of samples decline to fire"
    )


def test_shaping_and_gae_use_the_same_gamma():
    """A silent divergence here breaks policy invariance.

    The environment discounts the potential with Config.gamma and GAE
    discounts returns with PPOConfig.gamma. Ng/Harada/Russell requires the
    same gamma in both; if they drift apart the shaping is no longer
    potential-based and can change which policy is optimal, with nothing
    visibly failing.
    """
    from hockey.config import DEFAULT
    assert PPOConfig().gamma == pytest.approx(DEFAULT.gamma)


def test_possession_rate_sits_inside_its_viable_window():
    """The rate must beat the drag but not outscore a goal.

    Too low and carrying the puck still loses money to the discount drag; too
    high and hoarding for a whole episode beats scoring. At gamma 0.995 the
    window was empty -- this pins that it is both non-empty and respected.
    """
    import numpy as np
    from hockey.config import DEFAULT as C
    from hockey.env import VecHockeyEnv

    # Measure Phi in a real carrying state rather than hardcoding it, so this
    # keeps testing the true window when the weights are retuned.
    env = VecHockeyEnv(num_envs=1, seed=0)
    env.skater_pos[0] = [[-6.0, 0.0], [-27.0, 12.0]]
    env.theta[0] = [0.0, np.pi]
    env.omega[:] = 0.0
    env.skater_vel[:] = 0.0
    env.puck_pos[:] = [-6.0 + C.blade_offset, 0.0]
    env.puck_vel[:] = 0.0
    env._update_possession()
    assert env.possessor[0] == 0
    phi_while_holding = abs(float(env._potential()[0]))

    drag = (1 - C.gamma) * phi_while_holding
    hoard_ceiling = C.goal_reward / C.max_episode_steps
    assert drag < hoard_ceiling, (
        f"no viable rate exists: drag {drag:.5f} >= hoard ceiling {hoard_ceiling:.5f}"
    )
    assert drag < C.possession_rate < hoard_ceiling, (
        f"possession_rate {C.possession_rate} outside ({drag:.5f}, {hoard_ceiling:.5f})"
    )
    # Scoring must strictly dominate hoarding, not merely edge it out.
    assert C.possession_rate * C.max_episode_steps < 0.6 * C.goal_reward, (
        "a full-episode hold is worth too much next to a goal"
    )


def test_taking_a_shot_is_not_priced_out_of_reach():
    """The potential given up on release sets a minimum shot success rate.

    At 30.5% the policy learned to keep the puck and stop shooting entirely
    (shots fell to 1-2 per diagnostic and goals with them), which is the
    mirror image of the earlier failure where it shot on every step it held
    the puck. The bar has to sit between those.
    """
    from hockey.config import DEFAULT as C
    offensive = 0.6                              # (d_own - d_opp)/rink_length there
    cost = C.shaping_weight * offensive * (1 - C.loose_puck_factor) + C.possession_weight
    breakeven = cost / C.goal_reward
    assert 0.03 < breakeven < 0.20, f"break-even shot success rate is {breakeven:.1%}"


def test_checkpoint_round_trips_full_training_state(tmp_path):
    """A checkpoint must be enough to *resume*, not merely to play.

    This environment restarts containers roughly hourly, which killed a 30M
    step run at 3.7M with no error at all. Without optimizer state and the
    opponent pool, resuming silently restarts Adam's moments and throws away
    the frozen opponents, so a resumed run is not the run it claims to
    continue.
    """
    t = PPOTrainer(PPOConfig(num_envs=8, rollout_steps=8, pool_size=3, seed=0))
    for _ in range(2):
        b, _ = t.collect(); t.learn_on(b)
    t._snapshot(); t._snapshot()
    t.update = 7
    path = str(tmp_path / "latest.pt")
    t.save(path)

    fresh = PPOTrainer(PPOConfig(num_envs=8, rollout_steps=8, pool_size=3, seed=0))
    next_update = fresh.load_for_resume(path)

    assert next_update == 8
    assert fresh.global_step == t.global_step
    assert len(fresh.pool_nets) == len(t.pool_nets) == 2, "opponent pool must survive"
    for a, b_ in zip(t.net.parameters(), fresh.net.parameters()):
        assert torch.allclose(a, b_)
    assert np.allclose(fresh.norm.mean, t.norm.mean)
    # Adam moments, not just the weights.
    assert fresh.opt.state_dict()["state"].keys() == t.opt.state_dict()["state"].keys()
    assert len(fresh.opt.state_dict()["state"]) > 0, "optimizer state was empty"
