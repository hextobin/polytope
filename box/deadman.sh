#!/bin/bash
# Every 5 minutes, independent of the experiment. Self-destructs when the
# deadline passes or control/stop is pushed to main (the remote kill switch).
# If the experiment service has died without finishing, pulls the deadline in
# to one hour, so a crash costs at most an hour of idle GPU.
#
# To fix a crash by hand: restart exp-run.service first, then (if needed) write
# a later epoch to /root/deadline.
source /root/.exp.env
source /root/exp/box/lib.sh
exec >>"$LOGS/deadman.log" 2>&1
now=$(date +%s)

if [ "$now" -ge "$(cat "$DEADLINE")" ]; then
  exec "$EXP/box/self_destruct.sh" "deadline passed"
fi
if [ ! -e "$LOGS/DONE" ] && ! systemctl is-active -q exp-run.service; then
  pull_deadline_in $(( now + 3600 )) "experiment service stopped without finishing"
fi
cd "$EXP"
if git fetch -q origin main && git cat-file -e origin/main:control/stop 2>/dev/null; then
  exec "$EXP/box/self_destruct.sh" "control/stop pushed to main"
fi
