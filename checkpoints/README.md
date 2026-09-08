# Checkpoints

Training output lives in `runs/`, which is gitignored and container-local. These
three are committed instead, because the numbers quoted in the top-level README
are measured from them and are otherwise unreproducible.

| file | steps | what it is | best vs ChaseBot |
|---|---|---|---|
| `v0-baseline.pt` | 29.5M | original reward, unbounded policy mean | -7.73 /min |
| `v2-control-gated-reward.pt` | 19.7M | puck position discounted while loose | -6.66 /min |
| `v3-bounded-mean.pt` | 30.0M | + policy mean squashed into the action range | -6.98 /min |

`v3` is the strongest: it beats `v0` 67-39 head-to-head with ends swapped, and
ties `v2` (65-81, CI includes 0.5). None of them plays hockey well -- see the
top-level README for what is still broken.

## Use them

```bash
python -m hockey.diagnose --a checkpoints/v3-bounded-mean.pt --b chase
python -m hockey.evaluate --a checkpoints/v3-bounded-mean.pt --b checkpoints/v0-baseline.pt
python -m hockey.watch    --a checkpoints/v3-bounded-mean.pt --b chase --out game.gif
python -m hockey.play     --opponent checkpoints/v3-bounded-mean.pt
```

## Why v0 and v2 are here too

They are the evidence for the main finding of the project. The policy mean is
unbounded in v0 and v2 and squashed through tanh in v3, and the shoot dimension
shows the runaway directly:

| checkpoint | actor shoot mean while holding | sigma |
|---|---|---|
| v0 | +3.50 | 1.295 |
| v2 | +4.44 (still climbing) | 1.129 |
| v3 | +0.99 (at the tanh bound) | 1.000 |

The action space is `[-1, 1]`. Once the mean sits that far outside it every
sample executes identically, exploration in that dimension dies, and no reward
change can pull it back -- which is why fixing the reward in v2 changed nothing.

Reproduce with `python -m hockey.diagnose --a <checkpoint> --b <checkpoint>`.

## Loading them in code

```python
from rl.ppo import load_policy
policy, ck = load_policy("checkpoints/v3-bounded-mean.pt")
action = policy.act(obs)          # obs is (..., 34); action is (..., 3)
print(ck["global_step"], ck["ppo_config"]["bounded_mean"])
```

`load_policy` reads `bounded_mean` and `max_log_std` from the checkpoint and
defaults them to the pre-fix behaviour when absent, so v0 and v2 rehydrate
exactly as they trained rather than being silently reinterpreted under the new
network semantics.
