#!/bin/bash
# The experiment service: install Python packages, check the GPU, run
# box/experiment.sh, then finish up. Log: /root/logs/run.log
set -uo pipefail
source /root/.exp.env
source /root/exp/box/lib.sh
cd "$EXP"
mkdir -p "$LOGS" "$OUT"
exec >>"$LOGS/run.log" 2>&1

note "installing python packages: $PIP_PACKAGES"
# First boot often has unattended upgrades holding the apt lock; wait for it.
apt-get -qq -o DPkg::Lock::Timeout=900 update
apt-get -qq -o DPkg::Lock::Timeout=900 install -y python3-venv python3-dev >/dev/null
if ! { [ -x .venv/bin/python ] || python3 -m venv .venv; } ||
   ! .venv/bin/pip -q install $PIP_PACKAGES; then
  exec "$EXP/box/self_destruct.sh" "python setup failed"
fi
if ! .venv/bin/python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  nvidia-smi || true
  exec "$EXP/box/self_destruct.sh" "no GPU visible to torch"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv

note "starting experiment.sh"
if PY="$EXP/.venv/bin/python" bash "$EXP/box/experiment.sh"; then
  note "experiment finished"
  touch "$LOGS/DONE"
  "$EXP/box/sync.sh"
  if [ "${KEEP_AFTER_DONE_MIN:-0}" -gt 0 ]; then
    pull_deadline_in $(( $(date +%s) + KEEP_AFTER_DONE_MIN * 60 )) "done, keeping box for inspection"
  else
    exec "$EXP/box/self_destruct.sh" "experiment done"
  fi
else
  note "experiment.sh FAILED (exit $?)"
  # Leave an hour to fix it; the dead-man timer cleans up if nobody does.
  pull_deadline_in $(( $(date +%s) + 3600 )) "experiment failed"
fi
