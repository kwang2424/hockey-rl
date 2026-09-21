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
| exp-chase | `--set chase_opponent_prob=0.5` (was 0) | 12M | 0.069 | **loss** — barely above random (0.064) |
| exp-100m | *(no change)* -- 100M steps, snapshots every 10M | 70M (v6) | **0.579, beats v2 98-38** | **win** -- but all of it bought between 20M and 40M |
| exp-sigma | `--set max_log_std=0.0` (sigma <= 1, was uncapped) | 50M, 2 arms | tie at every milestone | **null** -- and the control varied more than the treatment |

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

### exp-chase: the structural hypothesis

The first change here that is not a knob on the reward or the optimiser.

The ladder's own numbers motivate it: ChaseBot sits at 0.96 goal share while no
learned policy has exceeded 0.45, and every learned run plateaued by roughly
10M steps. Self-play between two weak policies may simply produce no useful
gradient -- neither punishes the other's mistakes, so there is nothing to
climb. Training half the envs against a strong fixed opponent is the standard
remedy, and it costs no extra compute: the same steps, against something worth
beating.

`opp_id` now encodes -1 the learner, -2 ChaseBot, >=0 a pool snapshot.
ChaseBot-driven agents are masked out of the loss exactly as pool-driven ones
are, and `chase_opponent_prob` and `pool_prob` split the probability mass
rather than cannibalising each other.

### exp-100m: the plateau was a measurement artifact -- but compute ran out too

100M steps, one unmodified run, snapshots every 10M, every snapshot laddered
against a fixed field. Roughly three hours of free container time.

**Final ladder, 13 entrants, 78 pairs, both ends:**

| rank | entrant | share |
|---|---|---|
| 1 | chase | 0.899 |
| 2 | **70M** | **0.579** |
| 3 | 90M | 0.574 |
| 4 | 100M (final) | 0.559 |
| 5 | 80M | 0.556 |
| 6 | 60M | 0.530 |
| 7 | 50M | 0.512 |
| 8 | 40M | 0.468 |
| 9 | *v2 (previous best)* | *0.341* |
| 10 | 30M | 0.238 |
| 11 | 20M | 0.160 |
| 12 | 10M | 0.107 |
| 13 | random | 0.095 |

**Consecutive snapshots, played directly:**

| interval | result |
|---|---|
| 10M -> 20M | 4 - 5 tie |
| 20M -> 30M | 6 - 40 **clear gain** |
| 30M -> 40M | 32 - 67 **clear gain** |
| 40M -> 50M | 61 - 71 tie |
| 50M -> 60M | 77 - 76 tie |
| 60M -> 70M | 69 - 79 tie |
| 70M -> 80M | 80 - 77 tie |
| 80M -> 90M | 88 - 82 tie |
| 90M -> 100M | 77 - 88 tie |

All the skill in this run was bought between 20M and 40M. The sharpest way to
say it: **50M and 100M play 80-80.** The last half of the run bought nothing
measurable.

**vs v2, the fixed reference (margin):** -43, -43, -18, +23, +35, +62, +60,
+74, +71, +75. Widens through 60M, then plateaus.

#### The correction this run forced

The "v2 plateaued by 10M" reading, stated repeatedly in this file, was wrong,
and the way it was wrong is the lesson. It came from `vs_chase` going flat --
but ChaseBot was so much stronger than any learned policy that goal difference
saturated at about -9/min whether the policy was improving or not. The metric
had no resolution in the range where all the progress was happening. It only
came back into range late: chase's share over the field fell 0.955 -> 0.899 by
the end, and against 70M it manages 277-77 rather than the 326-32 it managed
against 40M.

Consequences, both acted on:

- **`best.pt` is selected on `vs_chase`**, so every experiment's "best"
  checkpoint was an arbitrary snapshot rather than its strongest. This run's
  own `best.pt` is a ~70M snapshot by luck, not design. Ladder `final.pt` and
  the archive explicitly; do not trust `best.pt`.
- **Ladder trajectories, not endpoints.** A single final number cannot tell a
  climb from a plateau, and the six-loss narrative below was built entirely on
  endpoints -- taken at 12M, which this run shows is *before the learning
  starts*. Those six verdicts are mostly measurements of noise. They are left
  recorded below because the reasoning is still worth reading, but they should
  not be treated as settled.

#### Where it actually stopped, and why

Policy standard deviation, read straight off each checkpoint -- free, no
simulation required:

| | std(fwd) | std(turn) | std(shoot) |
|---|---|---|---|
| 10M | 0.677 | 0.528 | 0.931 |
| 20M | 0.857 | 0.511 | 1.504 |
| 30M | 0.938 | 0.496 | 2.151 |
| 40M | 0.996 | **0.336** | 3.207 |
| 50M | 1.001 | 0.237 | 4.639 |
| 60M | 1.116 | 0.173 | 6.290 |
| 70M | 1.321 | 0.132 | 7.799 |
| 80M | 1.565 | 0.109 | 9.468 |
| 90M | 1.759 | 0.097 | 10.647 |
| 100M | 1.815 | **0.096** | **11.122** |

Ten points, monotonic in every column, no reversals.

`mean_reward` sits at roughly +/-0.0002 all run -- the shaping terms very nearly
cancel -- so on any action channel where the task gradient is weak, the entropy
bonus is the only force acting and it inflates sigma unopposed. `shoot` ends at
sigma = 11.1 against a [-1, 1] clip: a fair coin on every step. `fwd` follows
it. `turn` is the sole channel with a gradient strong enough to resist, and it
tightens 5.5x.

The timing lines up exactly. `std(turn)` sits flat at 0.53/0.51/0.50 through
30M, then breaks to 0.336 in the 30M -> 40M interval -- the same interval as
the one big ladder jump and the crossover past v2. The steering channel
starting to sharpen and the skill gain are the same event. After that turn
keeps tightening while the ladder crawls, i.e. the policy goes on refining
positioning it has already largely learned, while the half of the game that
needs the shoot channel stays out of reach.

**So compute was genuinely the binding constraint from 10M to 40M -- a 5x gain
in goal share that nothing else in this project produced -- and it stopped
being the binding constraint at 40M.** Both halves of that sentence are
results. More steps will not recover a channel whose gradient has already been
drowned.

The clean single-variable follow-up is a per-channel entropy floor, or simply a
lower `entropy_coef`. Note this is *not* a re-run of exp-entropy below, which
lowered the coefficient globally and was judged at 12M -- inside the range now
known to be all noise.

`checkpoints/v6-100m-compute.pt` is the 70M snapshot, the top-ranked entrant.
Verified after slimming: beats v2 96-45, ties its own source 82-67.

### exp-sigma: capping the action noise changed nothing measurable

The follow-up exp-100m pointed at: the entropy bonus was inflating sigma on
channels with no task gradient until the shoot trigger was a coin flip, so cap
it. `max_log_std` already existed, making this a genuine one-flag change. Two
fresh 50M-step arms, same seed, differing only in that flag.

**It did what it was designed to do.**

| shoot sigma | 10M | 20M | 30M | 40M | 50M |
|---|---|---|---|---|---|
| control (uncapped) | 0.820 | 1.181 | 1.613 | 1.961 | 2.141 |
| capped (sigma <= 1) | 0.809 | 1.001 | 1.001 | 1.001 | 1.001 |

Pinned from 20M on, while `turn` went on sharpening freely in both arms
(0.46 -> 0.30 control, 0.50 -> 0.32 capped). The cap bit only the channels that
were inflating, exactly as intended.

**And it bought nothing.** Same milestone, played head to head:

| milestone | control - capped |
|---|---|
| 10M | 40 - 11 *(before the cap binds)* |
| 20M | 29 - 34 |
| 30M | 49 - 44 |
| 40M | 41 - 48 |
| 50M | 69 - 65 |

Summed over every milestone where the cap is actually active: 188 - 191. Final
ratings -0.589 and -0.608. There is no effect here to argue about.

#### The control moved more than the treatment did

This is the part worth keeping. `exp-sigma-ctl` is the same config and the same
seed as `exp-100m`, on a different container. It behaved very differently:

| at 40M | exp-100m | exp-sigma-ctl |
|---|---|---|
| vs v2, head to head | **77 - 54 win** | **34 - 59 loss** |
| shoot sigma at 50M | 4.639 | 2.141 |

So the sigma inflation that motivated this whole experiment was less than half
as severe in the fresh control, and the skill trajectory was worse by a margin
far larger than the treatment effect being tested. Same knobs, same seed,
different box.

Two consequences:

- **This was a weaker test than intended.** The mechanism being corrected was
  mild in this control, so the cap had little to fix. It does not rule out the
  cap helping a run that inflates the way exp-100m did.
- **Single-run trajectories carry less weight than this file has been giving
  them.** The big exp-100m result -- goal share 0.107 -> 0.522 between 10M and
  40M -- is large enough to survive this much variance. The fine structure is
  not: the exact location of the bend, and the tidy story about `turn`
  sharpening in the same interval as the skill jump, sit inside the noise band
  that this comparison just measured. Treat them as one run's shape, not as
  established mechanism.

The honest next step is not another knob. It is running the *same* arm two or
three times and measuring the spread, so that future one-change experiments
have an error bar to be judged against. Every verdict in this file, wins and
losses alike, was recorded without one.

### Six for six

Every deliberate improvement since v2 has lost:

| change | verified to work mechanically? | ladder |
|---|---|---|
| puck-on-stick curriculum | yes | loss |
| bounded policy mean | yes | loss |
| gamma 0.995 -> 0.998 | yes | loss (in v5) |
| repriced shooting | yes | loss (in v5) |
| entropy_coef 0.004 -> 0.001 | yes | loss |
| ChaseBot as training opponent | yes | loss |

The pattern is not that the diagnoses were wrong. Each mechanism was measured
doing exactly what it was designed to do. The pattern is that **mechanism-level
correctness has had no predictive relationship with skill** in this
environment, and the only reliable signal has been head-to-head play.

The last entry mattered most, because it was the only one that was not a knob
on the reward or the optimiser. If self-play between two weak policies produced
no useful gradient, training half the envs against a far stronger fixed
opponent should have helped. It did not, which removes the most plausible
structural explanation on the table.

What remains untested is the one thing no change here addressed: whether
anything improves given an order of magnitude more steps. v2 plateaued by
roughly 10M of its 30M, which argues against it, but nothing has actually run
long enough to settle it. That test is free on this hardware -- about three
hours of uptime with `--resume` nudges -- and is the honest next step before
concluding the setup is at fault.

## Practical notes

- The container here restarts roughly hourly and kills training. `--resume`
  continues from `<out>/latest.pt` with optimizer state and opponent pool
  intact; checkpoints are written every update.
- **Do not run two training processes on this box.** Solo, an update takes
  1.3s (about 19k steps/sec). Two concurrent processes take 44s per update
  *each* -- a 30x collapse, not the 2x that sharing four cores would predict,
  and it does not depend on `OMP_NUM_THREADS`. It has the shape of OpenMP
  spin-wait barriers under thread oversubscription. A two-arm experiment
  planned as 90 minutes concurrent would in fact have taken about 25 hours;
  run the arms sequentially instead, which costs exactly the sum and nothing
  more.
- Single-seed results at 1M steps are noise. A reward change once looked like
  a 3.3x goal-rate win on seed 0 and vanished across three seeds.
- Short runs cannot test slow bugs. The mean-saturation failure took tens of
  millions of steps to appear; at 1M the bounded and unbounded runs were
  indistinguishable.
