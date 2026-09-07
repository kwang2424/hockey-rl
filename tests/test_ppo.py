"""Trainer-level regression tests."""

import numpy as np
import torch

from rl.ppo import PPOTrainer, PPOConfig
from rl.nets import RunningNorm
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
