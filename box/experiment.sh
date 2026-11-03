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
set -euo pipefail
source /root/exp/box/lib.sh

# Example: replace everything below.
note "data prep would go here"
progress_start

UNITS=(a b c)
for i in "${!UNITS[@]}"; do
  name=${UNITS[$i]}
  if [ ! -f "$OUT/$name.json" ]; then
    note "start $name"
    $PY -c "import torch, json; x = torch.randn(4096, 4096, device='cuda'); print(json.dumps({'norm': float((x @ x).norm())}))" > "$OUT/$name.json"
    note "done $name"
  fi
  progress $((i + 1)) ${#UNITS[@]}
done
