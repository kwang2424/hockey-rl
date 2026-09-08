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
python -m pytest -q                       # 50 tests, ~20s
python -m hockey.watch --a chase --b chase --out demo.gif

# Trained policies are committed in checkpoints/ -- no need to train first.
python -m hockey.diagnose --a checkpoints/v3-bounded-mean.pt --b chase
python -m hockey.play     --opponent checkpoints/v3-bounded-mean.pt

python train.py --total-steps 30000000 --out runs/v4
```

`runs/` is gitignored and container-local; the three checkpoints the results
below are measured from are committed under `checkpoints/` so every number here
is reproducible. See `checkpoints/README.md`.

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

### 5. Watch it in Godot (or Unity)

```bash
python -m hockey.export --a runs/v1/best.pt --b chase --out viewer/godot/game.json
godot --path viewer/godot
```

The sim is decoupled from rendering, so this does *not* mean porting the
physics into an engine — it exports a state trajectory and replays it, which
keeps training at full speed and lets the viewer be as pretty as you like. A
working Godot 4 project is in `viewer/godot/` (play/pause, frame-step, scrub,
speed control); `viewer/README.md` has the schema and a Unity script sketch.

### 6. Plot the curves

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
| **closer to the puck than your opponent** | 1.00 | the term that gets a fresh policy off the ground |

That third term is load-bearing, and its weight is the single most important
number in the file. Under a random policy the puck-position term contributes
~3.5x more per-step reward variance than the proximity term, and almost none
of that is yet controllable -- so at a low weight the one signal a fresh agent
*can* act on is buried in noise. Measured over 737k steps on the full reward:

| proximity weight | mean distance to puck (random = 14.21 m) |
|---|---|
| 0.25 | 13.29 -> 13.54 -> **13.57** (reverses) |
| 1.00 | 12.88 -> 12.24 -> **12.15** (monotonic) |

Because every term is potential-based, reweighting like this **cannot change
which policy is optimal** -- only what is learnable early. That is a genuinely
useful property to lean on: you can bias hard toward the controllable signal
without biasing the solution.

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

- **v1 (in progress) — the puck-on-stick curriculum.** A fraction of resets
  start with the puck already on a skater's blade, facing the net, with the
  defender goal-side; the rate anneals from 0.75 to 0.05 over training.
  Scoring is otherwise gated behind winning the puck, so a fresh policy sees
  almost no goals and cannot attribute one to anything it did. The curriculum
  applies to the *training* env only — `Config.puck_on_stick_prob` defaults
  to 0.0 so no curriculum-inflated number can leak into a reported score, and
  a test enforces that.
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

Honest status after four 30M-step runs: **two real bugs found and fixed, the
policy is measurably improving, and it is still nowhere near competent.**

| run | change | best vs ChaseBot | head-to-head |
|---|---|---|---|
| v0 | baseline | -7.73 /min | — |
| v1 | puck-on-stick curriculum | -9.19 /min | loses 5-47 to v0 |
| v2 | control-gated reward | -6.66 /min | — |
| v3 | + bounded policy mean | -6.98 /min | **beats v0 67-39**; ties v2 |

`hockey.evaluate` swaps ends and reports a CI, so "beats v0" means the interval
excludes 0.5 and "ties v2" means it does not (65-81, too close to call).

### The two bugs, and why behaviour beat curves at finding them

**Displacement was priced above control.** One second of shooting moved the
puck 30m (+0.1750 of shaping); one second of carrying moved it 11m (+0.0642)
plus the possession term. Firing on contact beat carrying by +0.031/s, so the
policy fired on 99% of the steps it held the puck, with 0 of 103 shots on
target. Fixed by discounting the position term while the puck is loose.

**The policy mean escaped the action range.** The Gaussian mean was an
unbounded `Linear` while the env clips actions to [-1, 1]. On the shoot
dimension it walked to +3.50 (and was still climbing -- v2 reached +4.42):

| checkpoint | shoot mean while holding | sigma |
|---|---|---|
| v0 | +3.496 | 1.295 |
| v2 | +4.416 | 1.129 |
| v3 (tanh-bounded) | **+0.990** | 1.000 |

At +3.5 with sigma 1.29, roughly 1 possession in 278 ever sampled "do not
shoot" -- and possessions lasted one step. Every sample executed identically,
so the environment could not distinguish the policy from a constant and
nothing pulled the mean back. **This is why fixing the reward changed nothing:
the reward was saying the right thing to a dimension that could no longer hear
it.** Squashing the mean through tanh took `fire_while_holding` off 1.0 for the
first time across every run and sweep in the project (1.0 -> 0.806), and
possessions finally exceeded a single step.

Neither bug is visible on a return curve. Both are obvious in one
`hockey.diagnose` call.

### Why possession_weight could never have worked

Potential-based shaping pays `F = gamma*Phi(s') - Phi(s)`, which means it pays
the *transition into* a state and charges the *transition out* -- it cannot
express "this is good per unit time". Tracing one possession at
`possession_weight = 0.20`:

| | reward |
|---|---|
| gain the puck | +0.200 |
| each step held | -0.001 (pure discount drag) |
| shoot / lose it | -0.200 |
| **net over 25 steps** | **-0.025** |

A longer possession pays strictly *less* than a short one. No value of
`possession_weight` can make the policy want to carry the puck, which is why
raising it 0.08 -> 0.20 changed nothing. Rewarding duration requires a term
outside the potential (`Config.possession_rate`), and that costs the
policy-invariance guarantee every other term here keeps.

That term has an unresolved tension worth knowing about:

```
discount drag while holding        -0.0030 /step
rate 0.005  ->  net +0.0020/step,  full-episode hoard = 3.00 vs a goal = 1.00
```

To keep hoarding the puck for a whole episode worth less than one goal needs
`rate < 0.00167` -- which is *below* the drag, so holding would still lose
money. There is no value that both beats the drag and prices below a goal.
The drag itself comes from `(1-gamma)*Phi`, so the real fix is probably to
shrink `Phi` (the proximity term at weight 1.0 dominates it and has already
done its job) rather than to keep tuning the rate. Until then the default is
0.005 and `hockey.diagnose` flags hoarding directly.

Measured effect at 1M steps, three seeds -- mean possession seconds:

| rate | per seed | mean |
|---|---|---|
| 0.0 | 0.057, 0.067, 0.044 | 0.056 |
| 0.005 | 0.113, 0.033, 0.050 | 0.065 |
| 0.015 | 0.147, 0.049, 0.033 | 0.076 |

The means trend the right way and goals/min rose monotonically (0.102 ->
0.141 -> 0.188), but seed 0 drives almost all of it. **Inconclusive at this
budget.** This is the third time a clean seed-0 result failed to replicate.

### Turning: a hard turn was a brake

Rendering scripted manoeuvres (`python -m hockey.drills --gif out.gif`) showed
turns at speed collapsing into tiny stalling curls rather than carving arcs.
Tracing the velocity through one:

| step | speed | forward | lateral |
|---|---|---|---|
| before the turn | 8.18 | 8.18 | 0.00 |
| +0.3s | 8.60 | 7.10 | 4.86 |
| +0.5s | 7.48 | 3.52 | 6.60 |
| +0.9s | 2.96 | **-1.82** | 2.33 |

The heading was free to spin at 4.5 rad/s while the grip budget supports only
`grip_accel_max / v` = 1.27 rad/s at 11 m/s. Heading that outruns the velocity
converts forward speed into *lateral* speed, and lateral speed is exactly what
the blade destroys -- so hard turns braked, and forward velocity went negative.

The turn rate is now capped at `min(max_omega, grip_accel_max / speed)`: you
cannot turn harder than your edges can hold. The same manoeuvre now keeps
speed (8.18 -> 10.98 m/s through the turn, lateral held at 0.70) and traces a
5.7 m arc. Pivoting on the spot is unaffected, since the cap relaxes to
`max_omega` below 3.1 m/s.

This is the bug a return curve could never have shown, and neither could the
numeric drill output -- `hard_left_at_speed` "passed" its heading check while
braking to a stop. It took looking at the path.

### A dead config parameter

`max_omega` was unreachable. The three turning parameters interact: under full
command the heading rate settles at `turn_accel / ang_damp`, and `max_omega`
only ever binds if that quotient exceeds it. It did not -- 16.0 / 5.5 settles
at 2.91 rad/s against a declared 4.5 cap, so the clip never fired, skaters
pivoted 35% slower than the config claimed, and tuning `max_omega` did nothing
at all.

`turn_accel` is now 28.0 so the cap actually binds, and a test asserts it stays
that way. A 180-degree pivot went from 1.23s to 0.80s.

Worth noting what that did *not* fix: reversing direction at speed got
*slower* (0.97s to 1.33s at 8 m/s), because there the binding constraint is the
grip budget, not the turn rate. At 11 m/s the tightest sustainable turn radius
is 8.6 m -- the heading can now swing faster than the blades can redirect the
velocity, so the skater slides. That is correct behaviour for a skate, and the
reason a real player slows down before turning hard.

### What is still broken

v3 still fails three of five behavioural checks: possession 0.041s (needs
>0.25s), 3.4% of shots on target, and a blade-vs-body gap of 0.05m, meaning it
closes on the puck without turning to face it. Its shoot mean sits pinned *at*
the tanh bound (+0.990), so it still wants to fire as hard as it is allowed --
the runaway is contained, not the preference. ChaseBot, for comparison, holds
the puck 0.78s and puts 80% of shots on target.

### Two things that cost real time

- **Single-seed results at 1M steps are noise.** A reward change looked like a
  3.3x goal-rate win on seed 0; three seeds erased it entirely. Budget three
  or more seeds, which triples the cost of the fast loop.
- **Short runs cannot test slow bugs.** The mean-saturation pathology takes
  tens of millions of steps to develop; at 1M the bounded and unbounded runs
  are indistinguishable (fire rate 0.59 vs 0.67). Match the experiment horizon
  to the timescale of what you are testing, or you will conclude nothing
  loudly.

## Roadmap

v0 is deliberately the smallest thing that proves the loop end to end.

- **v1 (in progress) — the puck-on-stick curriculum.** A fraction of resets
  start with the puck already on a skater's blade, facing the net, with the
  defender goal-side; the rate anneals from 0.75 to 0.05 over training.
  Scoring is otherwise gated behind winning the puck, so a fresh policy sees
  almost no goals and cannot attribute one to anything it did. The curriculum
  applies to the *training* env only — `Config.puck_on_stick_prob` defaults
  to 0.0 so no curriculum-inflated number can leak into a reported score, and
  a test enforces that.
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
- **Diagnose with an isolated reward before blaming the algorithm.** Two
  separate times, "it isn't learning" turned out to be a reward-balance
  problem rather than a broken trainer. Stripping the reward to a single
  controllable term and measuring one scalar (mean distance to the puck) is a
  60-second experiment that settles it.
- **Self-play evals are noisy.** 64 envs x 400 steps against `random` swings
  by several goals; the same checkpoint measured 0-4 and then 3-2 a million
  steps apart. Use `hockey.evaluate`, which swaps ends and gives you a CI,
  and treat `vs_chase` (hundreds of goals) as the real signal.

Where the compute should go next, in order: more steps, then a GPU with far
more parallel envs, then a curriculum that starts the puck on your stick so
scoring is discoverable before puck-winning is solved.
