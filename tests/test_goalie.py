"""Scripted goalie and solid net: both off by default, both on for new work."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from hockey.config import DEFAULT, Config
from hockey.env import (VecHockeyEnv, OBS_DIM, GOALIE_OBS_DIM, GOALIE_OBS_SLICES,
                        obs_dim_for, _rotate_into_body)
from hockey import rink

GOALIE = replace(DEFAULT, goalie=True, solid_net=True)
OPEN = replace(DEFAULT, goalie=False, solid_net=False)     # the original game


def _roll(cfg, steps=200, n=16, seed=4):
    env = VecHockeyEnv(num_envs=n, cfg=cfg, seed=seed)
    rng = np.random.default_rng(seed)
    out = [env.observe()]
    rews = []
    for _ in range(steps):
        obs, r, *_ = env.step(rng.uniform(-1, 1, (n, 2, 3)))
        out.append(obs)
        rews.append(r)
    return env, np.stack(out), np.stack(rews)


# ---------------------------------------------------------------- defaults
def test_default_is_the_goalie_game():
    assert DEFAULT.goalie is True and DEFAULT.solid_net is True
    assert obs_dim_for(DEFAULT) == OBS_DIM + GOALIE_OBS_DIM
    assert obs_dim_for(OPEN) == OBS_DIM


def test_open_game_is_bit_identical_to_the_original():
    """A fingerprint of 700 random-action steps of the open-net game, recorded
    with both flags off at commit b4b5ffa (the original game's code). If
    this fails after a numpy upgrade alone, regenerate it at b4b5ffa with the
    same code; if it fails after a sim change, that change leaked into the
    original game and every experiment before goalie-v9 no longer reproduces."""
    import hashlib
    env = VecHockeyEnv(32, OPEN, seed=5)
    rng = np.random.default_rng(1)
    h = hashlib.sha256()
    h.update(env.reset().tobytes())
    for _ in range(700):
        o, r, *_ = env.step(rng.uniform(-1, 1, (32, 2, 3)))
        h.update(o.tobytes()); h.update(r.tobytes()); h.update(env.puck_pos.tobytes())
    assert h.hexdigest() == "21ff64cb6c8d39d2e956537d485cb3fcba9c437bb8f27f8243c9a0d10093a046"


def test_goalie_knobs_do_nothing_while_the_goalie_is_off():
    """With goalie and solid_net off, no goalie or net parameter can reach the
    game: the base game stays exactly the one every earlier experiment ran."""
    _, o0, r0 = _roll(OPEN)
    tweaked = replace(OPEN, goalie_radius=2.0, goalie_max_speed=50.0,
                      goalie_reaction=0.5, goalie_skill_min=0.0, net_restitution=0.9)
    env, o1, r1 = _roll(tweaked)
    assert np.array_equal(o0, o1) and np.array_equal(r0, r1)
    assert o1.shape[-1] == OBS_DIM
    assert not hasattr(env, "goalie_pos")


# ---------------------------------------------------------------- goalie
def test_goalie_game_observation_appends_goalie_slots():
    env = VecHockeyEnv(num_envs=8, cfg=GOALIE, seed=0)
    for _ in range(20):
        env.step(np.random.default_rng(0).uniform(-1, 1, (8, 2, 3)))
    obs = env.observe()
    assert env.obs_dim == OBS_DIM + GOALIE_OBS_DIM == obs.shape[-1]
    L = GOALIE.rink_length
    # Agent a attacks net a, so goalie a is its attacking goalie.
    want_atk = _rotate_into_body(env.goalie_pos - env.skater_pos, env.theta) / L
    want_dfd = _rotate_into_body(env.goalie_pos[:, ::-1] - env.skater_pos, env.theta) / L
    np.testing.assert_allclose(obs[..., GOALIE_OBS_SLICES["atk_goalie_rel"]], want_atk, atol=1e-6)
    np.testing.assert_allclose(obs[..., GOALIE_OBS_SLICES["dfd_goalie_rel"]], want_dfd, atol=1e-6)


def test_goalie_never_leaves_its_crease_box():
    env, _, _ = _roll(GOALIE, steps=400, n=32)
    # Run a few more steps, checking every one.
    rng = np.random.default_rng(1)
    for _ in range(200):
        env.step(rng.uniform(-1, 1, (32, 2, 3)))
        out, lat = env._to_net(env.goalie_pos)
        assert (out >= GOALIE.goalie_min_out - 1e-9).all()
        assert (out <= GOALIE.goalie_max_out + 1e-9).all()
        assert (np.abs(lat) <= GOALIE.goalie_max_lateral + 1e-9).all()
        # Each goalie stays in front of its own net.
        assert (env.goalie_pos[:, 0, 0] > 0).all() and (env.goalie_pos[:, 1, 0] < 0).all()


def test_skaters_cannot_overlap_a_goalie():
    env, _, _ = _roll(GOALIE, steps=10, n=4)
    env.skater_pos[:, 0] = env.goalie_pos[:, 0] + [-0.3, 0.0]     # inside it
    env.skater_vel[:, 0] = [5.0, 0.0]
    env.step(np.zeros((4, 2, 3)))
    gap = np.linalg.norm(env.skater_pos[:, 0] - env.goalie_pos[:, 0], axis=-1)
    assert (gap >= GOALIE.skater_radius + GOALIE.goalie_radius - 0.2).all()


def test_rewards_stay_zero_sum_with_the_goalie():
    _, _, r = _roll(GOALIE, steps=300)
    np.testing.assert_array_equal(r[..., 1], -r[..., 0])


def test_goalie_saves_shots_at_it_and_loses_to_close_aimed_corners():
    """The calibration the goalie was tuned to (analysis/goalie_calibration):
    a shot straight at it never scores; a perfectly aimed corner from 6 m
    usually does; the same corner from 20 m does not."""
    from analysis.goalie_calibration import shoot
    assert shoot(GOALIE, 10.0, 0.0, 100, seed=1) == 0.0
    assert shoot(GOALIE, 6.0, 0.78, 100, seed=2) > 0.7
    assert shoot(GOALIE, 20.0, 0.78, 100, seed=3) < 0.1
    # And without the goalie the same far corner shot goes in.
    assert shoot(replace(GOALIE, goalie=False), 20.0, 0.78, 100, seed=3) > 0.8


def test_goalie_skill_is_drawn_per_episode_within_range():
    env = VecHockeyEnv(num_envs=256, cfg=GOALIE, seed=0)
    lo, hi = GOALIE.goalie_skill_min, GOALIE.goalie_skill_max
    assert (env.goalie_skill >= lo).all() and (env.goalie_skill <= hi).all()
    assert env.goalie_skill.std() > 0.1
    env.goalie_skill_range = (0.0, 0.0)          # curriculum: a static goalie
    env.reset()
    assert (env.goalie_skill == 0.0).all()
    before = env.goalie_pos.copy()
    env.step(np.zeros((256, 2, 3)))
    np.testing.assert_array_equal(env.goalie_pos, before)


# ---------------------------------------------------------------- solid net
def _puck_only_env(cfg, pos, vel):
    env = VecHockeyEnv(num_envs=len(pos), cfg=cfg, seed=0)
    env.skater_pos[:] = [[-20.0, 6.0], [-20.0, -6.0]]
    env.skater_vel[:] = 0.0
    env.possessor[:] = -1
    env.cooldown[:] = 1e9
    env.puck_pos[:] = pos
    env.puck_vel[:] = vel
    if cfg.goalie:
        env._reset_goalies(np.arange(len(pos)))
    return env


def _goals(env, steps=40):
    scored = np.zeros(env.n, dtype=bool)
    live = np.ones(env.n, dtype=bool)
    for _ in range(steps):
        _, _, _, _, info = env.step(np.zeros((env.n, 2, 3)))
        scored |= live & (info["goal_a"] | info["goal_b"])
        live &= ~info["episode_end"]
        env.cooldown[:] = 1e9
    return scored


def test_open_net_counts_pucks_from_behind_and_solid_net_does_not():
    """The hole the learned agents were exploiting: a puck travelling along
    the end boards' side of the net, across the back of it, used to score."""
    gx = DEFAULT.goal_line_x
    pos = np.array([[gx + 1.5, 4.0], [gx + 0.6, -4.0], [gx + 2.0, 3.0]])
    vel = np.array([[0.0, -15.0], [0.0, 15.0], [-6.0, -12.0]])     # behind / beside
    assert _goals(_puck_only_env(OPEN, pos, vel)).all()
    solid = replace(OPEN, solid_net=True)
    assert not _goals(_puck_only_env(solid, pos, vel)).any()


def test_solid_net_still_scores_through_the_mouth():
    gx = DEFAULT.goal_line_x
    solid = replace(OPEN, solid_net=True)
    pos = np.array([[gx - 8.0, 0.0], [gx - 6.0, 2.0], [-gx + 5.0, -1.0]])
    aim = np.array([[gx, 0.5], [gx, 0.3], [-gx, -0.6]]) - pos
    vel = 25.0 * aim / np.linalg.norm(aim, axis=-1, keepdims=True)
    assert _goals(_puck_only_env(solid, pos, vel)).all()


def test_solid_net_keeps_puck_and_skaters_out_of_the_box():
    cfg = replace(DEFAULT, solid_net=True)
    env, _, _ = _roll(cfg, steps=10, n=64)
    rng = np.random.default_rng(3)
    gx, gh, D = cfg.goal_line_x, cfg.goal_half_width, cfg.goal_depth
    for _ in range(400):
        env.step(rng.uniform(-1, 1, (64, 2, 3)))
        for sgn in (1.0, -1.0):
            u = (env.skater_pos[..., 0] - sgn * gx) * sgn
            v = env.skater_pos[..., 1]
            inside = (u > 0.05) & (u < D - 0.05) & (np.abs(v) < gh - 0.05)
            assert not inside.any()


def test_crossed_goal_line_needs_the_front_and_the_mouth():
    cfg = DEFAULT
    gx, gh = cfg.goal_line_x, cfg.goal_half_width
    prev = np.array([[gx - 0.2, 0.0], [gx - 0.2, gh + 0.3], [gx + 0.5, 0.0], [-gx + 0.2, 0.1]])
    pos = np.array([[gx + 0.2, 0.0], [gx + 0.2, gh + 0.3], [gx + 0.9, 0.0], [-gx - 0.2, 0.1]])
    plus, minus = rink.crossed_goal_line(cfg, prev, pos)
    assert plus.tolist() == [True, False, False, False]
    assert minus.tolist() == [False, False, False, True]


# ---------------------------------------------------------------- networks
def test_old_checkpoint_plays_the_goalie_game_blind_to_the_goalies():
    from rl.ppo import load_policy
    pol, _ = load_policy("checkpoints/v8-screened-50m.pt")
    assert pol.obs_dim == OBS_DIM
    env = VecHockeyEnv(num_envs=8, cfg=GOALIE, seed=0)
    obs = env.observe()[:, 0]
    np.testing.assert_array_equal(pol.act(obs), pol.act(obs[:, :OBS_DIM]))


def test_widened_checkpoint_acts_exactly_like_the_original():
    from rl.ppo import PPOTrainer, PPOConfig, load_policy
    from train import init_from_checkpoint
    path = "checkpoints/v8-screened-50m.pt"
    ck = torch.load(path, weights_only=False)
    t = PPOTrainer(PPOConfig(num_envs=4, rollout_steps=8, **{
        k: ck["ppo_config"][k] for k in ("hidden", "bounded_mean", "max_log_std")
        if k in ck["ppo_config"]}), GOALIE)
    assert t.env.obs_dim == OBS_DIM + GOALIE_OBS_DIM
    init_from_checkpoint(t, path)
    old, _ = load_policy(path)
    env = VecHockeyEnv(num_envs=16, cfg=GOALIE, seed=5)
    for _ in range(30):
        env.step(np.random.default_rng(0).uniform(-1, 1, (16, 2, 3)))
    obs = env.observe()[:, 0]
    np.testing.assert_allclose(t.policy().act(obs), old.act(obs), atol=1e-5)
    # And the new inputs are wired in, ready to learn.
    assert t.net.actor[0].weight.shape[1] == OBS_DIM + GOALIE_OBS_DIM


def test_trainer_builds_its_network_for_the_goalie_game():
    from rl.ppo import PPOTrainer, PPOConfig
    t = PPOTrainer(PPOConfig(num_envs=4, rollout_steps=8, num_minibatches=2), GOALIE)
    assert t.net.actor[0].weight.shape[1] == OBS_DIM + GOALIE_OBS_DIM
    assert t.norm.mean.shape == (OBS_DIM + GOALIE_OBS_DIM,)
    batch, _ = t.collect()
    assert batch["obs"].shape[-1] == OBS_DIM + GOALIE_OBS_DIM
    t.learn_on(batch)
    assert all(torch.isfinite(q).all() for q in t.net.parameters())


def test_sniper_beats_the_goalie_where_chasebot_cannot():
    from hockey.bots import ChaseBot, SniperBot
    from hockey.rollout import play
    sn, _ = play(SniperBot(GOALIE), SniperBot(GOALIE), num_envs=64, steps=600, seed=2, cfg=GOALIE)
    ch, _ = play(ChaseBot(GOALIE), ChaseBot(GOALIE), num_envs=64, steps=600, seed=2, cfg=GOALIE)
    assert sn["goals_a"] + sn["goals_b"] > 5 * (ch["goals_a"] + ch["goals_b"] + 1)


def test_goalie_curriculum_ramps_the_training_goalie_only():
    from rl.ppo import PPOTrainer, PPOConfig
    t = PPOTrainer(PPOConfig(num_envs=4, rollout_steps=8, goalie_curriculum_steps=1000), GOALIE)
    t.global_step = 0
    assert t.set_goalie_curriculum() == 0.0 and t.env.goalie_skill_range == (0.0, 0.0)
    t.global_step = 500
    assert t.set_goalie_curriculum() == 0.5
    assert t.env.goalie_skill_range == pytest.approx((GOALIE.goalie_skill_min / 2,
                                                      GOALIE.goalie_skill_max / 2))
    t.global_step = 5000
    assert t.set_goalie_curriculum() == 1.0
    # An eval env is built from the Config: full skill regardless.
    t.global_step = 0
    t.set_goalie_curriculum()
    ev = VecHockeyEnv(num_envs=4, cfg=t.env.cfg, seed=0)
    assert ev.goalie_skill_range == (GOALIE.goalie_skill_min, GOALIE.goalie_skill_max)
    # Off (the default) never touches the env.
    off = PPOTrainer(PPOConfig(num_envs=4, rollout_steps=8), GOALIE)
    assert off.set_goalie_curriculum() == 1.0
    assert off.env.goalie_skill_range == (GOALIE.goalie_skill_min, GOALIE.goalie_skill_max)


def test_evaluate_plays_the_game_being_trained():
    from rl.ppo import PPOTrainer, PPOConfig
    from train import evaluate, resolve_anchors
    import hockey.rollout as ro, train as tr
    seen = []
    real = tr.play
    def spy(*a, **k):
        seen.append(k.get("cfg"))
        return real(*a, **k)
    tr.play = spy
    try:
        t = PPOTrainer(PPOConfig(num_envs=4, rollout_steps=8), GOALIE)
        evaluate(t, 5, 2, resolve_anchors(["sniper", "random"]))
    finally:
        tr.play = real
    assert seen and all(c is GOALIE for c in seen)
