"""Actor-critic network and observation normaliser."""

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2.0), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def mlp(in_dim, out_dim, hidden=(128, 128), out_std=0.01):
    layers, last = [], in_dim
    for h in hidden:
        layers += [layer_init(nn.Linear(last, h)), nn.Tanh()]
        last = h
    layers += [layer_init(nn.Linear(last, out_dim), std=out_std)]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Gaussian policy with a state-independent log-std, plus a value head.

    Actor and critic are separate trunks; with a 34-dim observation the extra
    parameters cost nothing and it avoids the value loss dominating the
    shared features early on.
    """

    def __init__(self, obs_dim, act_dim, hidden=(128, 128), init_log_std=-0.5,
                 bounded_mean=True, max_log_std=0.0):
        super().__init__()
        self.actor = mlp(obs_dim, act_dim, hidden, out_std=0.01)
        self.critic = mlp(obs_dim, 1, hidden, out_std=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))
        self.bounded_mean = bounded_mean
        self.max_log_std = max_log_std

    def dist(self, obs):
        """Gaussian over actions, with the mean squashed into the action range.

        The squash is load-bearing, not cosmetic. The environment clips actions
        to [-1, 1], so an unbounded mean can walk outside that range and the
        executed action stops responding to it -- every sample clips to the
        same value, exploration in that dimension dies, and the policy can no
        longer discover the alternative even when the reward says it should.

        That is exactly what happened here: the shoot mean reached +3.48 with
        sigma 1.29, so roughly 1 possession in 278 ever sampled "do not
        shoot", and possessions last a single step. The policy fired on 100%
        of the steps it held the puck across three separate 30M-step runs, and
        fixing the reward could not budge it, because the reward signal had
        nothing left to act on.
        """
        mean = self.actor(obs)
        if self.bounded_mean:
            mean = torch.tanh(mean)
        log_std = self.log_std.clamp(max=self.max_log_std)
        return torch.distributions.Normal(mean, log_std.exp().expand_as(mean))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def act(self, obs, deterministic=False):
        d = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        return a, d.log_prob(a).sum(-1), self.value(obs)

    def evaluate(self, obs, action):
        d = self.dist(obs)
        return d.log_prob(action).sum(-1), d.entropy().sum(-1), self.value(obs)


class RunningNorm:
    """Welford running mean/variance for observation whitening."""

    def __init__(self, shape, clip=10.0, eps=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps
        self.clip = clip

    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).astype(np.float64)
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var = (m_a + m_b + delta**2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x, update=False):
        if update:
            self.update(x)
        z = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count, "clip": self.clip}

    def load_state_dict(self, sd):
        self.mean = np.asarray(sd["mean"])
        self.var = np.asarray(sd["var"])
        self.count = float(sd["count"])
        self.clip = float(sd.get("clip", 10.0))


class TorchPolicy:
    """Adapter so an ActorCritic satisfies the same act(obs) API as the bots."""

    def __init__(self, net: ActorCritic, norm: RunningNorm, device="cpu"):
        self.net = net
        self.norm = norm
        self.device = device

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        shape = obs.shape[:-1]
        flat = self.norm(obs.reshape(-1, obs.shape[-1]))
        t = torch.as_tensor(flat, device=self.device)
        a, _, _ = self.net.act(t, deterministic=deterministic)
        return np.clip(a.cpu().numpy().reshape(*shape, -1), -1.0, 1.0)
