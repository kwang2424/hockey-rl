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
| 1 | `chase` (scripted) | 0.962 |
| 2 | **`v2-control-gated-reward`** | **0.449** |
| 3 | `v3-bounded-mean` | 0.333 |
| 4 | `v0-baseline` | 0.217 |
| 5 | `v5-repriced-shooting` | 0.076 |
| 6 | `random` | 0.072 |
| 7 | `exp-ent001` (entropy_coef 0.001) | 0.070 |
| 8 | `exp-bounded` (bounded_mean on) | 0.056 |

Shares shift slightly as entrants are added, since each plays everyone; the
ordering is what matters.

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

### Baseline hygiene: build on the incumbent, not on the latest code

Before the first experiment, the defaults had to be moved back. They had
drifted to v5's configuration -- gamma 0.998, `loose_puck_factor` 0.75,
`possession_weight` 0.08, `proximity_weight` 0.6, `possession_rate` 0.0008,
`bounded_mean` on -- which the ladder ranks at 0.063, barely above random.
"One change against the incumbent" run from there would silently have been
"v5 plus one change".

Defaults now match v2 on every reward and PPO knob. Two deliberate exceptions:

- `turn_accel` stays at 28.0. That is a physics correctness fix, not a tuning
  choice: below it a hard turn braked a skater from 8.2 m/s to 1.4 m/s with
  forward velocity going negative, which `hockey/drills.py` demonstrates
  directly. v2 is *evaluated* under the fixed physics and still wins.
- `bounded_mean` and `possession_rate` remain implemented but default off, so
  each is one `--set` away from being tested properly.

Three tests had to be rewritten, because they asserted beliefs the ladder
contradicted -- most starkly one requiring the break-even shot rate to be
under 20%, when the 30.5% configuration turned out to be the best learned
policy and the 13.2% one nearly the worst. They now pin mechanisms and record
quantities rather than asserting that a particular value is correct.

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
| exp-ent001 | `--set entropy_coef=0.001` (was 0.004) | 12M | 0.070 | **loss** — below random (0.072) |
| exp-bounded | `--set bounded_mean=1` (was off) | 12M | 0.056 | **loss** — last, below random |

Append a row per experiment. Record the losses; they are the entries that
changed how this project was run.

### exp-ent001, in full

The reasoning: entropy climbed in every previous run (v0 3.40, v3 3.59,
v5 3.16), which is the signature of the entropy bonus outrunning the policy
gradient and the policy diffusing back toward random. Lower the coefficient
and it should hold still.

The mechanism worked exactly as designed. Entropy fell monotonically for the
first time in the project, 2.64 -> 1.88, reversing the pattern in every prior
run.

The player came last, below `random`.

That is the fourth consecutive time a well-argued change, verified to do the
thing it was designed to do, produced a worse hockey player. The most likely
reading is the obvious one: less exploration meant the policy committed early
to a poor strategy and never left it. Diagnosing the *mechanism* correctly
said nothing about whether changing it helps.

### exp-bounded, in full

Isolates the one change that separated v2 from v3: squashing the policy mean
through tanh so it cannot leave the action range. The bug it fixes is real and
was measured directly -- v0's actor emitted a shoot mean of +3.50 against an
action range of [-1, 1], so about 1 possession in 278 ever sampled "do not
shoot" and exploration in that dimension was effectively dead.

It came **last**, at 0.056, below random and below the unbounded incumbent.

This settles the earlier ambiguity in the least comfortable direction. v3 was
v2 plus this change and scored worse; now the change on its own, cleanly
isolated, scores worse again. So it was not something else in v3 -- bounding
the mean genuinely costs more than the saturation it prevents, at least at
this scale.

The uncomfortable implication is that the saturated mean was doing useful
work. An actor pinned far outside the action range produces a near-deterministic
action in that dimension, and "always shoot on contact" appears to be a better
policy at this skill level than anything the agent finds when it retains the
freedom to choose.

### Five for five

Every deliberate improvement since v2 has lost:

| change | verified to work mechanically? | ladder |
|---|---|---|
| puck-on-stick curriculum | yes | loss |
| bounded policy mean | yes | loss |
| gamma 0.995 -> 0.998 | yes | loss (in v5) |
| repriced shooting | yes | loss (in v5) |
| entropy_coef 0.004 -> 0.001 | yes | loss |

The pattern is not that the diagnoses were wrong. Each mechanism was measured
doing exactly what it was designed to do. The pattern is that **mechanism-level
correctness has had no predictive relationship with skill** in this
environment, and the only reliable signal has been head-to-head play.

## Practical notes

- The container here restarts roughly hourly and kills training. `--resume`
  continues from `<out>/latest.pt` with optimizer state and opponent pool
  intact; checkpoints are written every update.
- Single-seed results at 1M steps are noise. A reward change once looked like
  a 3.3x goal-rate win on seed 0 and vanished across three seeds.
- Short runs cannot test slow bugs. The mean-saturation failure took tens of
  millions of steps to appear; at 1M the bounded and unbounded runs were
  indistinguishable.
