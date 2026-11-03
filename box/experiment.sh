#!/bin/bash
# THE ONLY BOX SCRIPT YOU EDIT PER EXPERIMENT. Run by run.sh on the droplet,
# from the repo root, with $PY pointing at the venv's python.
#
# Rules:
# - Write everything worth keeping to $OUT (pushed to the results branch).
#   Keep files under 95MB; compress or bf16 checkpoints if needed.
# - Make units resumable: skip work whose output already exists, so a restart
#   of exp-run.service picks up where it left off.
# - Call progress_start after setup/data prep, and `progress i n` after each
#   unit, so the deadline tightens to the real runtime.
# - Exit non-zero on failure. The box then gives you an hour to fix it.
#
# polytope: 5 arms x 3 seeds, 5000 steps each, on enwik8 (90/10 split, as in
# polydeq). Each run is one unit; a run with final.json is skipped. Order is all
# arms at seed 0, then seed 1, then seed 2, so a partial sweep stays balanced.
set -euo pipefail
source /root/exp/box/lib.sh
cd "$EXP"

ARMS=(floor box l1 shrink ceiling)
SEEDS=(0 1 2)
DATA=$EXP/data/enwik8

if [ ! -f "$DATA/val.bin" ]; then
  note "downloading enwik8"
  mkdir -p "$EXP/data"
  curl -fsSL --retry 5 --retry-delay 10 -o "$EXP/data/enwik8.zip" http://mattmahoney.net/dc/enwik8.zip
  $PY -m zipfile -e "$EXP/data/enwik8.zip" "$EXP/data/raw"
  size=$(stat -c %s "$EXP/data/raw/enwik8")
  [ "$size" -eq 100000000 ] || { note "enwik8 has $size bytes, expected 100000000"; exit 1; }
  $PY -c "from polytope.data import prepare_text; prepare_text('$EXP/data/raw/enwik8', '$DATA')"
fi
note "data ready"

# Quick correctness check of the projections before spending GPU time.
$PY - <<'EOF'
import torch
from polytope.proj import l1_tau, proj_l1
z = torch.randn(1024, 384, device="cuda")
y = proj_l1(z, 96.0)
assert (y.abs().sum(-1) <= 96.0 + 1e-3).all()
assert torch.allclose(y.abs().sum(-1), torch.full((1024,), 96.0, device="cuda"), atol=1e-2)
print("proj check ok, nonzero frac", (y != 0).float().mean().item())
EOF
progress_start

total=$(( ${#ARMS[@]} * ${#SEEDS[@]} ))
i=0
for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    i=$((i + 1))
    run=$OUT/runs/${arm}_s${seed}
    if [ ! -f "$run/final.json" ]; then
      note "start $arm seed $seed ($i/$total)"
      mkdir -p "$run"
      $PY train.py --data "$DATA" --out "$run" --arm "$arm" --seed "$seed" \
        --steps 5000 --batch_size 64 --block_size 256 --n_layer 6 --n_head 6 --d_model 384 \
        --eval_every 500 --compile > "$run/stdout.txt" 2>&1 \
        || { note "train.py failed for $arm seed $seed"; tail -30 "$run/stdout.txt"; exit 1; }
      note "done $arm seed $seed: $(tail -1 "$run/stdout.txt")"
    fi
    progress $i $total
  done
done

$PY scripts/summarize.py "$OUT/runs" --md "$OUT/summary.md"
