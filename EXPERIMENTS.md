# Experiment protocol

Two rules, both bought with wasted compute.

## 1. Rank by head-to-head play, not by metrics

```bash
python -m hockey.ladder --envs 192 --steps 600 --json ladder.json
```

Head-to-head is the only measurement in this project that ever contradicted a
confident conclusion. Behavioural metrics, return curves and per-step reward
arithmetic all agreed with each other and were wrong together.

Current standings (192 envs x 600 steps per side, both ends played):

| rank | entrant | goal share |
|---|---|---|
| 1 | `chase` (scripted) | 0.952 |
| 2 | **`v2-control-gated-reward`** | **0.378** |
| 3 | `v3-bounded-mean` | 0.256 |
| 4 | `v0-baseline` | 0.169 |
| 5 | `v5-repriced-shooting` | 0.063 |
| 6 | `random` | 0.053 |

Read that ordering carefully, because it is the opposite of the story the
diagnostics told:

- **v0 -> v2 helped** (0.169 -> 0.378). Gating the position term by puck
  control was a real improvement.
- **v2 -> v3 hurt** (0.378 -> 0.256). Bounding the policy mean fixed a genuine
  and independently verifiable bug -- the actor's shoot output had run to
  +3.50 against an action range of [-1, 1] -- and still made the player worse.
- **v3 -> v5 hurt badly** (0.256 -> 0.063). Raising gamma and repricing four
  weights left it barely above random.

**v2 is the incumbent.** Beat it or the change is not an improvement.

## 2. One change per run

```bash
python train.py --set entropy_coef=0.001 --total-steps 12000000 --out runs/exp-ent001
python -m hockey.ladder --add runs/exp-ent001/best.pt --envs 192 --steps 600
```

`--set` takes exactly one knob, routes it to whichever of `Config` or
`PPOConfig` declares it, **fails loudly on an unknown key**, and records the
override in the run's `config.json`. A silently ignored override would mean a
run labelled as testing a change actually tested nothing.

Three reward changes were stacked between v3 and v5. The composite is a
regression and isolating the culprit would now cost three more runs. That is
the whole reason for this rule.

`gamma` is special-cased: it is set on both `Config` (which discounts the
potential) and `PPOConfig` (which discounts returns), because a divergence
stops the shaping being potential-based with nothing visibly failing.

## Log

| run | single change vs incumbent | steps | goal share | verdict |
|---|---|---|---|---|
| v2 | *(incumbent)* | 19.7M | 0.378 | — |

Append a row per experiment. Record the losses; they are the entries that
changed how this project was run.

## Practical notes

- The container here restarts roughly hourly and kills training. `--resume`
  continues from `<out>/latest.pt` with optimizer state and opponent pool
  intact; checkpoints are written every update.
- Single-seed results at 1M steps are noise. A reward change once looked like
  a 3.3x goal-rate win on seed 0 and vanished across three seeds.
- Short runs cannot test slow bugs. The mean-saturation failure took tens of
  millions of steps to appear; at 1M the bounded and unbounded runs were
  indistinguishable.
