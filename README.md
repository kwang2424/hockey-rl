# hockey-rl

A 1v1 ice hockey environment and a PPO self-play agent, built to watch what
strategies fall out of a reward function.

Everything is a headless 2D sim in numpy. There is no game engine anywhere in
the training loop — that is the single most important decision in the project.
A Unity/Godot-backed env gives you a few thousand steps per second; this gives
you **~55,000 control steps/sec on one CPU core**, which is the difference
between "trained overnight" and "trained over a month". The sim emits state,
and a completely separate renderer replays it.

## Why hockey is a *good* fit for this (not a hard one)

The usual worry is that hockey mechanics are harder to model than soccer. They
are mostly easier:

- **Skating is easier than running.** Soccer envs fake locomotion with capsules
  pushed by forces, because legs are hard. Skating needs no fake: a skate is a
  rigid body with *anisotropic friction* — low along the blade, high across it.
  That is ~10 lines, and it isn't an approximation of skating, it essentially
  **is** skating. Momentum, carving, and sliding out of a hard turn all fall
  out for free.
- **A puck is easier than a ball.** It's 2D. No aerial trajectory, no bounce,
  no spin or Magnus effect.
- **The rink is enclosed.** Play never stops for out-of-bounds, so you get
  denser experience per second of wall clock.

The genuinely hard part is the stick, so v0 doesn't have one yet (see
[Roadmap](#roadmap)).

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest -q                       # 33 tests, ~40s
python -m hockey.watch --a chase --b chase --out demo.gif
python train.py --total-steps 20000000 --out runs/v0
```

## How to test it

Five independent ways, roughly in order of how much they tell you per minute
spent.

### 1. Play against it yourself

```bash
python -m hockey.play --opponent runs/v0/best.pt
```

`W`/`S` skate, `A`/`D` turn, `Space` shoots, `Tab` hands your side to the
agent so you can just watch, `R` resets the faceoff. You are red, attacking the
right-hand net.

Thirty seconds of this tells you more than any training curve. You immediately
feel whether the agent tracks the puck, whether it can be beaten by simply
skating around it, and whether it does anything that looks like defending.

### 2. Watch a game as a GIF (works headless)

```bash
python -m hockey.watch --a runs/v0/best.pt --b chase --out game.gif
python -m hockey.watch --a runs/v0/best.pt --b runs/v0/best.pt --out selfplay.gif
```

No display or game engine needed, so this works over SSH or in CI. The puck
carrier gets a yellow halo, which makes possession battles legible.

### 3. Measure it against fixed baselines

```bash
python -m hockey.evaluate --a runs/v0/best.pt --b chase
python -m hockey.evaluate --a runs/v0/best.pt --b runs/v0/latest.pt --envs 256
```

This is the part people skip and shouldn't. **Self-play return is close to
meaningless on its own** — a policy getting better and a policy getting worse
both hover around zero, because each is always playing something exactly as
good as itself. So progress is measured against fixed opponents:

| baseline | what it proves |
|---|---|
| `random` | the floor — anything should beat it |
| `still`  | isolates whether the sim itself is fair |
| `chase`  | a real bar: chases, carries to the net, shoots, plays goal-side |

`evaluate` plays both ends and swaps sides to cancel any residual side bias,
and reports a **95% confidence interval on goal share** plus a `decisive` flag.
Small evals swing by a couple of goals and look like progress; if it says
`too close to call`, widen `--envs`/`--steps` rather than believing it.

### 4. Run the test suite

```bash
python -m pytest -q
```

33 tests. The ones that matter most:

- **`test_nothing_escapes_the_rink`** / **`test_no_tunnelling_at_max_shot_speed`** —
  fires the puck at the boards from 360 angles at max speed.
- **`test_skating_is_anisotropic`** — the blade must kill lateral velocity far
  faster than forward glide. That ratio *is* the skating model.
- **`test_hard_turn_at_speed_slides_out`** — grip is a finite budget, so
  turning hard at speed must cost you.
- **`test_shaped_return_is_path_independent`** — a 68m route to the goal must
  pay *bit-identically* the same as a 62m route. This is what proves the reward
  has no farmable cycle.
- **`test_reward_is_exactly_zero_sum`** and the 180°-mirror symmetry tests.
- **`test_learn_on_ragged_batch_stays_finite`** — a real regression (see below).

Why so many physics tests: **an RL agent will find and exploit any physics bug
you leave in, and it will look like emergent behaviour until you check.** These
pin down what must be true no matter what the policy does.

### 5. Plot the curves

```bash
python plot_progress.py runs/v0/progress.csv
```

Writes a PNG (no matplotlib). Also read `runs/v0/progress.csv` directly — it
has per-update entropy, KL, gradient norm, and the baseline evals.

## Design notes

### Skating

```
lateral friction accel = min(lateral_damp * |v_lateral|, grip_accel_max)
```

The `min` is the whole trick. Below the cap the blade holds an edge and you
carve; above it you exceed the grip budget and slide. One line, and hard turns
at speed stop being free.

### Observations (34 dims, egocentric)

Team B's world is rotated **180°**, not mirrored in x. A mirror flips
chirality, so a policy that learned to turn left would have to unlearn it when
it swaps ends; a rotation doesn't. One policy therefore plays both sides, and
`test_env.py` checks the sim is bit-exactly symmetric under it.

Index observations through `OBS_SLICES` rather than hardcoding offsets.

### Reward

Zero-sum: `reward[:, 1] == -reward[:, 0]`, exactly, enforced by a test.

All shaping is **potential-based** (Ng, Harada & Russell 1999):

```
F = gamma * Phi(s') - Phi(s)
```

which is provably policy-invariant — it cannot change which policy is optimal,
so it cannot be farmed. This matters concretely: a naive per-step "the puck
moved toward the goal" bonus creates an oscillation exploit where the agent
rocks the puck back and forth forever. `Phi` has three terms, all antisymmetric
between the teams so the zero-sum property survives:

| term | weight | why |
|---|---|---|
| puck position between the nets | 0.35 | the actual objective |
| possession | 0.08 | the first thing worth learning |
| **closer to the puck than your opponent** | 0.25 | the term that gets a fresh policy off the ground |

That third term is load-bearing. Goals and possession are both far too rare for
a random policy to bootstrap from, but "skate at the puck harder than the other
guy" has a gradient on literally the first step.

### Self-play

One shared policy plays both sides, so every rollout yields two transitions per
env. A fraction of envs (`--pool-prob`) face a **frozen snapshot** of an older
policy instead of the current one. This is the part that matters for stability:
training only against your current self lets strategies cycle — A beats B beats
C beats A — and the policy chases its own tail forever. Agents driven by a pool
opponent are masked out of the loss.

## Bugs found while building this

Kept because they're the failure modes this kind of project actually hits:

- **A fixed `[1, 0]` fallback normal.** When a skater was deep in the rink
  interior the board-normal was undefined, and the code returned a fixed
  world-space direction. That silently broke the 180° team symmetry the entire
  self-play setup rests on — and it was a discontinuous observation feature
  besides. Replaced with a continuous, equivariant boundary vector.
- **A one-sample minibatch turned the whole network to NaN.** The learner mask
  makes the batch size ragged, so a strided `range(0, n, mb)` could leave a
  final minibatch of exactly one sample. Its *unbiased* `std()` is NaN, which
  divides into the advantages and poisons every weight. Fixed with
  `np.array_split`, plus a guard that skips non-finite gradients.
- **The boards weren't a hard constraint.** Puck-vs-body collisions resolved
  *after* the boards, so a puck pinned between a skater and the boards got
  squeezed 1cm outside the rink. Reordered so the boards always get the last
  word.

## Roadmap

v0 is deliberately the smallest thing that proves the loop end to end.

- **v1 — the stick.** A rigid segment instead of a fixed blade point. The real
  design crux is possession: a magnetic capture radius (what v0 does, and what
  is learnable) versus pure collision physics where carrying means repeated
  taps (much more interesting emergent behaviour, much harder to learn).
- **v2 — a goalie.** It is a genuinely different problem: it sees far fewer
  learning signals than a skater and will lag badly if trained as the same
  policy. Hand-code it first, then train it separately.
- **v3 — 2v2 / 3v3**, with a proper opponent league and a team-spirit parameter
  blending individual and team reward.
- **v4 — rules.** Offside and icing last, on purpose: offside is non-Markovian
  (it depends on the order in which players entered the zone), so it needs
  state the observation doesn't currently carry.

## A calibration note

The famous emergent-behaviour results (hide-and-seek box-surfing) came from
enormous compute. At this scale, expect something more modest.

But note which half is hard: **reward hacking is not the difficult part, it's
the default.** You get it for free in the first run. Competent play is the hard
part. And whatever emerges is emergent behaviour *of this sim*, not of hockey —
any inaccuracy is something the agents will find and live inside, which is why
`tests/test_physics.py` is as long as it is.

## What actually happens when you train it

Honest status: **v0 learns, measurably, but is not yet competitive with
`ChaseBot`.** On 4 CPU cores it needs more compute than a single sitting.

The decisive check is a stripped-down task — reward is *only* the
puck-proximity term, no goals, no possession bonus — which isolates "can this
setup learn anything at all?" from "is hockey hard?":

| steps | mean distance to puck | possession |
|---|---|---|
| random policy | 13.67 m | 0.001 |
| 246k | 11.61 m | 0.012 |
| 492k | 10.59 m | 0.013 |
| 737k | 9.89 m | 0.035 |
| 983k | 9.36 m | 0.031 |

Monotonic, and possession up ~30x from random. The machinery works; the full
game just needs far more of it. Reproduce with the config in
`PPOConfig(pool_prob=0.0)` and a `replace(DEFAULT, goal_reward=0.0,
shaping_weight=0.0, possession_weight=0.0, proximity_weight=1.0)` env.

Two things worth knowing before you burn a weekend on this:

- **Watch the gradient-step budget, not the env-step count.** At
  `num_envs=256, rollout_steps=96, num_minibatches=8` each update spends
  24,576 env steps on only 32 gradient steps, so 2.5M steps is ~3,200
  gradient steps — nowhere near enough to conclude anything. Raising
  `--num-minibatches` to 32 quadruples the gradient steps but was *slower* in
  wall-clock here (128 tiny matmuls each pay 4-thread sync overhead) and no
  better per sample at 246k steps. Left at 8.
- **Self-play evals are noisy.** 64 envs x 400 steps against `random` swings
  by several goals; the same checkpoint measured 0-4 and then 3-2 a million
  steps apart. Use `hockey.evaluate`, which swaps ends and gives you a CI,
  and treat `vs_chase` (hundreds of goals) as the real signal.

Where the compute should go next, in order: more steps, then a GPU with far
more parallel envs, then a curriculum that starts the puck on your stick so
scoring is discoverable before puck-winning is solved.
