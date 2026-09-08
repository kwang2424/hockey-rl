"""PPO with self-play and a frozen-opponent pool.

Self-play here is parameter sharing: one policy plays both sides, made
possible by the 180-degree team canonicalisation in the observation. Both
agents' transitions go into the buffer, so a rollout yields twice the data.

The opponent pool is the part that actually matters for stability. Training
only against the *current* policy lets strategies cycle -- A beats B beats C
beats A -- and the policy chases its own tail forever. Sampling frozen past
snapshots for some fraction of the envs forces the policy to stay good against
everything it used to be, not just against itself right now.
"""

import copy
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from hockey.config import Config, DEFAULT
from hockey.env import VecHockeyEnv, OBS_DIM, ACT_DIM
from .nets import ActorCritic, RunningNorm, TorchPolicy


@dataclass
class PPOConfig:
    num_envs: int = 256
    rollout_steps: int = 96
    total_steps: int = 3_000_000

    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.004
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 8
    target_kl: float = 0.03

    hidden: tuple = (128, 128)
    init_log_std: float = -0.5
    bounded_mean: bool = True    # squash the policy mean into the action range
    max_log_std: float = 0.0     # ceiling on exploration noise (sigma <= 1.0)

    # opponent pool
    pool_prob: float = 0.35      # fraction of envs facing a frozen snapshot
    pool_size: int = 8
    snapshot_every: int = 10     # updates

    eval_every: int = 10
    eval_steps: int = 400
    eval_envs: int = 64

    # Puck-on-stick curriculum, annealed on the *training* env only. Eval
    # always builds its own env from the Config (default 0.0), so a
    # curriculum-inflated score can never leak into a reported number.
    # Off by default: measured *worse* than no curriculum (a curriculum run
    # lost 5-47 head-to-head to an otherwise identical run without it). It
    # trains on states that do not occur under the evaluation distribution,
    # and the policy did not recover after the anneal. Kept because it is
    # worth retrying once possession actually works; opt in with
    # --curriculum-start.
    curriculum_start: float = 0.0
    curriculum_end: float = 0.0
    curriculum_frac: float = 0.6   # fraction of training spent annealing

    seed: int = 0
    device: str = "cpu"
    out_dir: str = "runs/v0"


class PPOTrainer:
    def __init__(self, ppo: PPOConfig = None, cfg: Config = DEFAULT):
        self.p = ppo or PPOConfig()
        self.cfg = cfg
        torch.manual_seed(self.p.seed)
        np.random.seed(self.p.seed)
        torch.set_num_threads(max(1, os.cpu_count() or 1))

        self.device = torch.device(self.p.device)
        self.env = VecHockeyEnv(self.p.num_envs, cfg=cfg, seed=self.p.seed)
        self.net = ActorCritic(OBS_DIM, ACT_DIM, self.p.hidden, self.p.init_log_std,
                               self.p.bounded_mean, self.p.max_log_std).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.p.lr, eps=1e-5)
        self.norm = RunningNorm(OBS_DIM)

        self.pool = []                                    # frozen state_dicts
        self.pool_nets = []
        self.rng = np.random.default_rng(self.p.seed + 777)
        # -1 means "team B is the learner too"; >=0 indexes into the pool.
        self.opp_id = np.full(self.p.num_envs, -1, dtype=np.int64)

        self.global_step = 0
        self.update = 0
        self.history = []

    # ------------------------------------------------------------------
    def policy(self):
        return TorchPolicy(self.net, self.norm, self.device)

    def set_curriculum(self, progress: float):
        """Anneal the puck-on-stick rate. ``progress`` runs 0 -> 1 over training."""
        p = self.p
        frac = min(1.0, max(0.0, progress / max(p.curriculum_frac, 1e-9)))
        value = p.curriculum_start + (p.curriculum_end - p.curriculum_start) * frac
        self.env.curriculum_puck_on_stick = float(value)
        return value

    def _snapshot(self):
        sd = copy.deepcopy({k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()})
        net = ActorCritic(OBS_DIM, ACT_DIM, self.p.hidden, self.p.init_log_std,
                          self.p.bounded_mean, self.p.max_log_std)
        net.load_state_dict(sd)
        net.eval()
        self.pool.append(sd)
        self.pool_nets.append(net)
        if len(self.pool) > self.p.pool_size:
            self.pool.pop(0)
            self.pool_nets.pop(0)

    def _reassign_opponents(self, idx):
        """Pick team B's controller for the envs that just reset."""
        if idx.size == 0 or not self.pool_nets:
            self.opp_id[idx] = -1
            return
        use_pool = self.rng.random(idx.size) < self.p.pool_prob
        picks = self.rng.integers(0, len(self.pool_nets), idx.size)
        self.opp_id[idx] = np.where(use_pool, picks, -1)

    @torch.no_grad()
    def _pool_actions(self, obs_b_norm):
        """Actions for team B in envs assigned to a frozen snapshot."""
        act = np.zeros((self.p.num_envs, ACT_DIM), dtype=np.float32)
        for i, net in enumerate(self.pool_nets):
            m = self.opp_id == i
            if not m.any():
                continue
            t = torch.as_tensor(obs_b_norm[m], device=self.device)
            a, _, _ = net.act(t, deterministic=False)
            act[m] = a.cpu().numpy()
        return np.clip(act, -1.0, 1.0)

    # ------------------------------------------------------------------
    def collect(self):
        """One rollout. Returns flattened tensors plus a learner mask."""
        p, n_env = self.p, self.p.num_envs
        T = p.rollout_steps

        obs_buf = np.zeros((T, n_env, 2, OBS_DIM), dtype=np.float32)
        act_buf = np.zeros((T, n_env, 2, ACT_DIM), dtype=np.float32)
        logp_buf = np.zeros((T, n_env, 2), dtype=np.float32)
        val_buf = np.zeros((T, n_env, 2), dtype=np.float32)
        rew_buf = np.zeros((T, n_env, 2), dtype=np.float32)
        done_buf = np.zeros((T, n_env), dtype=np.float32)     # true terminal only
        trunc_buf = np.zeros((T, n_env), dtype=np.float32)
        mask_buf = np.zeros((T, n_env, 2), dtype=np.float32)  # learner-controlled?
        tval_buf = np.zeros((T, n_env, 2), dtype=np.float32)   # V(s') at truncation

        ep_goals, ep_count = np.zeros(2), 0

        for t in range(T):
            raw = self.env.observe()
            norm = self.norm(raw.reshape(-1, OBS_DIM), update=True).reshape(n_env, 2, OBS_DIM)
            obs_buf[t] = norm

            flat = torch.as_tensor(norm.reshape(-1, OBS_DIM), device=self.device)
            with torch.no_grad():
                a, lp, v = self.net.act(flat)
            a = a.cpu().numpy().reshape(n_env, 2, ACT_DIM)
            logp_buf[t] = lp.cpu().numpy().reshape(n_env, 2)
            val_buf[t] = v.cpu().numpy().reshape(n_env, 2)

            learner_b = self.opp_id < 0
            if not learner_b.all():
                a[:, 1] = np.where(learner_b[:, None], a[:, 1], self._pool_actions(norm[:, 1]))
            act_buf[t] = a
            mask_buf[t, :, 0] = 1.0
            mask_buf[t, :, 1] = learner_b

            _, rew, goal, trunc, info = self.env.step(np.clip(a, -1.0, 1.0))
            rew_buf[t] = rew
            done_buf[t] = goal
            trunc_buf[t] = trunc

            if trunc.any():
                fo = self.norm(info["final_obs"][trunc].reshape(-1, OBS_DIM))
                with torch.no_grad():
                    fv = self.net.value(torch.as_tensor(fo, device=self.device))
                tval_buf[t, trunc] = fv.cpu().numpy().reshape(-1, 2)

            ended = info["episode_end"]
            if ended.any():
                ep_goals += [info["goal_a"].sum(), info["goal_b"].sum()]
                ep_count += int(ended.sum())
                self._reassign_opponents(np.nonzero(ended)[0])

        with torch.no_grad():
            last_raw = self.env.observe()
            last = self.norm(last_raw.reshape(-1, OBS_DIM)).reshape(n_env, 2, OBS_DIM)
            last_v = self.net.value(
                torch.as_tensor(last.reshape(-1, OBS_DIM), device=self.device)
            ).cpu().numpy().reshape(n_env, 2)

        adv, ret = self._gae(rew_buf, val_buf, done_buf, trunc_buf, last_v, tval_buf)
        self.global_step += T * n_env

        batch = dict(
            obs=obs_buf.reshape(-1, OBS_DIM),
            act=act_buf.reshape(-1, ACT_DIM),
            logp=logp_buf.reshape(-1),
            adv=adv.reshape(-1),
            ret=ret.reshape(-1),
            val=val_buf.reshape(-1),
            mask=mask_buf.reshape(-1),
        )
        stats = {
            "ep_count": ep_count,
            "goals_per_ep": float(ep_goals.sum() / max(ep_count, 1)),
            "mean_reward": float(rew_buf[..., 0].mean()),
            "pool_frac": float((self.opp_id >= 0).mean()),
        }
        return batch, stats

    def _gae(self, rew, val, done, trunc, last_v, tval):
        """Generalised advantage estimation.

        A goal is a true terminal (no bootstrap). A time-limit truncation is
        *not* -- the game did not really end, so we bootstrap through it.
        Conflating the two teaches the policy that the clock ending is a bad
        outcome to be avoided.
        """
        p = self.p
        T, n_env, _ = rew.shape
        adv = np.zeros_like(rew)
        last_gae = np.zeros((n_env, 2), dtype=np.float32)
        for t in reversed(range(T)):
            nonterminal = 1.0 - done[t][:, None]
            next_v = last_v if t == T - 1 else val[t + 1]
            # On truncation the stored next_v belongs to a fresh episode, so
            # substitute the value of the real pre-reset final observation.
            next_v = np.where(trunc[t][:, None] > 0, tval[t], next_v)
            reset = np.maximum(done[t], trunc[t])[:, None]
            delta = rew[t] + p.gamma * next_v * nonterminal - val[t]
            last_gae = delta + p.gamma * p.gae_lambda * (1.0 - reset) * last_gae
            adv[t] = last_gae
        return adv, adv + val

    # ------------------------------------------------------------------
    def learn_on(self, batch):
        p = self.p
        dev = self.device
        keep = np.nonzero(batch["mask"] > 0)[0]
        obs = torch.as_tensor(batch["obs"][keep], device=dev)
        act = torch.as_tensor(batch["act"][keep], device=dev)
        old_logp = torch.as_tensor(batch["logp"][keep], device=dev)
        adv = torch.as_tensor(batch["adv"][keep], device=dev)
        ret = torch.as_tensor(batch["ret"][keep], device=dev)

        n = obs.shape[0]
        idx = np.arange(n)
        info = {"n_train": n, "nonfinite_skips": 0}

        for epoch in range(p.update_epochs):
            np.random.shuffle(idx)
            # np.array_split, not a strided range: the learner mask makes the
            # batch size ragged, and a strided range can leave a final
            # minibatch of a single sample. Its unbiased std() is NaN, which
            # divides into the advantages and turns the whole network to NaN.
            for chunk in np.array_split(idx, p.num_minibatches):
                if chunk.size < 2:
                    continue
                b = torch.as_tensor(chunk, device=dev)
                logp, ent, v = self.net.evaluate(obs[b], act[b])
                ratio = (logp - old_logp[b]).exp()

                a = adv[b]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(a * ratio,
                                a * ratio.clamp(1 - p.clip_coef, 1 + p.clip_coef)).mean()
                vloss = 0.5 * (v - ret[b]).pow(2).mean()
                loss = pg + p.value_coef * vloss - p.entropy_coef * ent.mean()

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(self.net.parameters(), p.max_grad_norm)
                # Belt and braces: never let a single non-finite gradient
                # silently destroy every weight in the network.
                if not torch.isfinite(gnorm):
                    self.opt.zero_grad(set_to_none=True)
                    info["nonfinite_skips"] += 1
                    continue
                self.opt.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (logp - old_logp[b])).mean().item()
                info.update({"pg_loss": pg.item(), "v_loss": vloss.item(),
                             "entropy": ent.mean().item(), "approx_kl": approx_kl,
                             "grad_norm": float(gnorm)})
            if p.target_kl is not None and info.get("approx_kl", 0) > p.target_kl:
                info["stopped_epoch"] = epoch + 1
                break
        return info

    # ------------------------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "net": self.net.state_dict(),
            "norm": self.norm.state_dict(),
            "ppo_config": self.p.__dict__,
            "env_config": self.cfg.to_dict(),
            "global_step": self.global_step,
            "update": self.update,
        }, path)
        return path


def load_policy(path, device="cpu"):
    """Rehydrate a saved checkpoint into a drop-in Policy."""
    ck = torch.load(path, map_location=device, weights_only=False)
    hidden = tuple(ck["ppo_config"].get("hidden", (128, 128)))
    net = ActorCritic(OBS_DIM, ACT_DIM, hidden,
                      ck["ppo_config"].get("init_log_std", -0.5),
                      ck["ppo_config"].get("bounded_mean", False),
                      ck["ppo_config"].get("max_log_std", 10.0))
    net.load_state_dict(ck["net"])
    net.eval()
    norm = RunningNorm(OBS_DIM)
    norm.load_state_dict(ck["norm"])
    return TorchPolicy(net, norm, device), ck
