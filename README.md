# hockey-rl

A 1v1 ice hockey simulator and a PPO self-play agent, built to see what
strategies fall out of reinforcement learning.

The sim is a headless 2D numpy model, vectorised over hundreds of games at
once (~13k control steps/sec for training on 4 CPU cores). There is no game
engine in the training loop. The sim emits state and separate viewers replay
it.

**Where things stand** (details in `EXPERIMENTS.md`): the game is now
correct -- a solid net and a scripted goalie, so only close, aimed shots
score. No reinforcement-learning run has yet learned to score in it (28 of 28
from scratch or from v8, and 4 of 4 fine-tuned from an imitation of
SniperBot). The best network in this game is `checkpoints/bc-sniper.pt`, a
behaviour-cloned copy of the scripted SniperBot. The earlier learned agents
(v0-v8) were trained in an open-net game whose net had no back, and 90% of
their goals exploited that hole.

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest -q                                   # ~120 tests, a few minutes

python -m hockey.watch --a sniper --b chase --out demo.gif
python -m hockey.play  --opponent checkpoints/bc-sniper.pt   # needs pygame

python train.py --total-steps 50000000 --out runs/mine
```

`runs/` is gitignored and container-local. Checkpoints that results are
measured from are committed in `checkpoints/` (see `checkpoints/README.md`).

## The game

- **Rink**: 60 x 26 m, rounded corners, physics at 120 Hz, control at 30 Hz,
  20 s episodes.
- **Skating**: thrust along the heading only, and anisotropic friction --
  lateral velocity is killed hard up to a finite grip budget, so a hard turn at
  speed slides. That one `min()` is the whole skating model.
- **Stick and puck**: no articulated stick. A blade point sits 0.85 m ahead of
  each skater; a puck within the capture radius is carried, and the shoot
  action releases it along the heading at 14-30 m/s.
- **Net** (`Config.solid_net`, on): a box with a back and sides. A goal is the
  puck crossing the goal line between the posts from the front.
- **Goalie** (`Config.goalie`, on): one scripted goalie per net. It stands on
  the line from its net to the puck and slides onto the shot line when a shot
  comes, sees the puck 0.2 s late, has a top speed, and gets a random skill
  level per episode. Calibration (`analysis/goalie_calibration.py`): a
  perfectly aimed corner shot scores ~90% of the time from 6 m, 24% from 10 m
  and never from 14 m or more; a shot at the goalie never scores.
- **Rewards**: zero-sum (`r[:, 1] == -r[:, 0]`, tested). +1 per goal plus
  potential-based shaping (puck position, possession, being nearer the puck),
  which cannot change the optimal policy. One optional non-potential term,
  `proximity_rate`, for getting fresh runs to the puck.
- **Observations**: 38 numbers per skater, egocentric, team-canonical (team B
  sees the rink rotated 180 degrees, so one policy plays both ends). Index
  them through `OBS_SLICES` and `GOALIE_OBS_SLICES` in `hockey/env.py`.
- **Optional rule**: `min_hold_time` (settle before shooting).

`Config(solid_net=False, goalie=False)` is the original open-net game,
bit-identical to the one every experiment before goalie-v9 ran in.

## Layout

| path | what |
|---|---|
| `hockey/env.py`, `rink.py`, `config.py` | the simulator |
| `hockey/bots.py` | ChaseBot, SniperBot (aims at the open corner), RandomBot, StandStillBot |
| `hockey/measure.py` | standard seeded measurements: self-play, head to head, shot geometry, rating |
| `hockey/ladder.py`, `rating.py`, `diagnose.py`, `evaluate.py` | round robins, the anchored rating, behaviour stats |
| `hockey/seedspread.py`, `armcompare.py` | seed-variance and arm-vs-arm statistics |
| `hockey/watch.py`, `render.py`, `play.py`, `drills.py`, `export.py` | GIFs, play it yourself, physics drills, trajectory export |
| `rl/ppo.py`, `nets.py` | PPO with self-play, an opponent pool, and exact resume |
| `rl/bc.py` | behaviour cloning from a scripted player |
| `train.py` | training entry point; `--set KEY=VALUE` overrides any config field |
| `analysis/` | the goalie calibration; `archive/` holds each finished experiment's pre-registered script |
| `viewer/` | Game Tape (`build_tape.py`, a self-contained HTML replay page) and a Godot replay project |
| `results/` | raw outputs of past ladders and screens |
| `EXPERIMENTS.md` | protocol, every experiment and its verdict, corrections |

## Training

```bash
python train.py --total-steps 50000000 --seed 17 --set init_seed=203 --out runs/x
python train.py ... --resume                  # continue exactly from runs/x/latest.pt
python train.py ... --stop-at 3000000         # pause early, resume later (same --total-steps)
python train.py ... --init-from checkpoints/bc-sniper.pt --lr 1e-4    # fine-tune
```

One shared policy plays both sides. 35% of games are against frozen
snapshots of earlier selves (the opponent pool), which stops strategies
cycling. Every 15 updates the learner is rated against fixed anchors
(SniperBot, ChaseBot, random, bc-sniper) in a fresh copy of the training game,
and the best-rated snapshot is kept as `best.pt`.

Practical notes, each learned the hard way (more in `EXPERIMENTS.md`):

- **Run one training process at a time.** Two concurrent processes on this
  box are ~30x slower each, not 2x.
- **Seed variance is huge.** Identical configs span a goal share of 0.2-0.9,
  set mostly by the starting weights. Compare arms with several seeds and a
  permutation test, never one run each.
- **Pre-register.** Every experiment's analysis script is committed before
  its runs, with its pass/fail rule written down.
- **Measure behaviour, not curves.** Every plateau so far has been a bug or a
  reward error that one behavioural number (possession time, where goals go
  in, shot distance) exposed immediately.

## Watching

```bash
python -m hockey.watch --a checkpoints/bc-sniper.pt --b sniper --out game.gif
PYTHONPATH=. python viewer/build_tape.py viewer/tape/reels.json   # Game Tape page
python -m hockey.export --a sniper --b chase --out viewer/godot/game.json
```

The Game Tape (`viewer/game-tape.html`) is a self-contained page with
annotated reels, including the net-hole exploit and the goalie game.
