#!/usr/bin/env bash
# Runs the bc-finetune screen (analysis/bc_finetune.py). Restart-safe.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${1:-runs/bc-ft}
for s in $(seq 30 33); do
  d="$ROOT/s$s"; [ -f "$d/final.pt" ] && continue
  python3 train.py --total-steps 10000000 --seed $s --lr 1e-4 \
    --set solid_net=1 --set goalie=1 --init-from checkpoints/bc-sniper.pt \
    --eval-anchors sniper,chase,random,checkpoints/v8-screened-50m.pt --out "$d" --resume
done
[ -f "$ROOT/result.log" ] || { PYTHONPATH=. python3 analysis/bc_finetune.py "$ROOT" | tee "$ROOT/result.log.tmp"; mv "$ROOT/result.log.tmp" "$ROOT/result.log"; }
echo PIPELINE_DONE
