# Checkpoints

Training output lives in `runs/`, which is gitignored and container-local. These
three are committed instead, because the numbers quoted in the top-level README
are measured from them and are otherwise unreproducible.

| file | steps | what it is | best vs ChaseBot |
|---|---|---|---|
| `v0-baseline.pt` | 29.5M | original reward, unbounded policy mean | -7.73 /min |
| `v2-control-gated-reward.pt` | 19.7M | puck position discounted while loose | -6.66 /min |
| `v3-bounded-mean.pt` | 30.0M | + policy mean squashed into the action range | -6.98 /min |
| `v6-100m-compute.pt` | 70.0M | **no change at all** -- same setup, more steps | -5.16 /min |
| `v7-seed4-50m.pt` | 50.0M | same config again, seed 4 -- the strongest of twelve runs | -4.98 /min |

`v6` is by a clear margin the strongest, and the only one that beats `v2`
head-to-head (98-38). Nothing about the reward, network or hyperparameters
differs from the runs above; the only input was compute. It is the 70M snapshot
of `exp-100m`, chosen because it ranked top of a 13-entrant ladder over that
run's whole trajectory -- note that 50M through 100M are all statistically tied
with it, so the specific snapshot is close to arbitrary within that band.

Among the older three, `v3` is strongest: it beats `v0` 67-39 with ends
swapped, and ties `v2` (65-81, CI includes 0.5).

`v7` beats `v6` 97-70, and that is **not** evidence it is better. One pairing
at 0.581 goal share sits inside the band two runs of the *same* config produce
([0.215, 0.909], measured in exp-seeds), so the honest reading is "comparable,
at half the steps". It is kept because it was the best of the twelve runs of
exp-curric6 by goal share against that field, and because `runs/` is
container-local and would otherwise be lost -- not because it won anything.

None of them plays hockey well. `v6` scores by possession and crashing the net,
not by shooting -- its shoot channel is pure noise (sigma = 7.8 against a
[-1, 1] clip). See `EXPERIMENTS.md` for why.

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

## A note on physics drift

These were trained before two turning fixes: `max_omega` was unreachable dead
config, and the turn rate was not capped by the grip budget, so a hard turn
acted as a brake (8.2 m/s down to 1.4 m/s in a second) instead of a carve.
They therefore learned to drive a slightly different vehicle than the one in
`main` now.

They remain valid as evidence for the mean-saturation finding, which is a
property of the network rather than the physics, and that is what they are kept
for. Do not read their *play* as representative of the current sim.
