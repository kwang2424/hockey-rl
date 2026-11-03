#!/usr/bin/env bash
# Runs the goalie-v9 pipeline (see analysis/goalie_v9.py). Safe to re-run
# after a container restart: finished stages are skipped and an interrupted
# run continues exactly with --resume. One training process at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${1:-runs/goalie-v9}
COMMON=(--total-steps 50000000 --seed 17
        --set solid_net=1 --set goalie=1 --set goalie_curriculum_steps=15000000
        --eval-anchors sniper,chase,random,checkpoints/v8-screened-50m.pt)
mkdir -p "$ROOT"

for i in $(seq 400 411); do
  d="$ROOT/net$i"
  [ -f "$d/stage1.done" ] && continue
  python3 train.py "${COMMON[@]}" --set init_seed=$i --out "$d" --stop-at 3000000 --resume
  touch "$d/stage1.done"
done

if [ ! -f "$ROOT/selected.txt" ]; then
  PYTHONPATH=. python3 analysis/goalie_v9.py select "$ROOT" | tee "$ROOT/select.log"
fi

sel=$(cat "$ROOT/selected.txt")
d="$ROOT/net$sel"
if [ ! -f "$d/final.pt" ]; then
  python3 train.py "${COMMON[@]}" --set init_seed=$sel --out "$d" --resume
fi

if [ ! -f "$ROOT/final.log" ]; then
  PYTHONPATH=. python3 analysis/goalie_v9.py final "$ROOT" | tee "$ROOT/final.log.tmp"
  mv "$ROOT/final.log.tmp" "$ROOT/final.log"
fi
echo PIPELINE_DONE
