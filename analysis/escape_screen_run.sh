#!/usr/bin/env bash
# Runs the escape screen (analysis/escape_screen.py). Restart-safe; one
# training process at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${1:-runs/escape}
COMMON=(--total-steps 3000000 --set solid_net=1 --set goalie=1
        --set goalie_curriculum_steps=2000000
        --eval-anchors sniper,chase,random,checkpoints/v8-screened-50m.pt)
for i in $(seq 500 507); do
  d="$ROOT/chances/s$i"; [ -f "$d/final.pt" ] && continue
  python3 train.py "${COMMON[@]}" --seed 17 --set init_seed=$i \
    --curriculum-start 0.5 --curriculum-end 0.0 --set proximity_rate=0.003 --out "$d" --resume
done
for s in $(seq 20 27); do
  d="$ROOT/warm/s$s"; [ -f "$d/final.pt" ] && continue
  python3 train.py "${COMMON[@]}" --seed $s --lr 1e-4 \
    --init-from checkpoints/v8-screened-50m.pt --out "$d" --resume
done
[ -f "$ROOT/result.log" ] || { PYTHONPATH=. python3 analysis/escape_screen.py "$ROOT" | tee "$ROOT/result.log.tmp"; mv "$ROOT/result.log.tmp" "$ROOT/result.log"; }
echo PIPELINE_DONE
