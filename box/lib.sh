# Shared paths and helpers. Sourced by the box scripts and by experiment.sh.
# EXP_STATE exists only so the scripts can be tested off the box.
S=${EXP_STATE:-/root}
EXP=$S/exp
LOGS=$S/logs
DEADLINE=$S/deadline
OUT=$EXP/out          # experiment.sh writes everything worth keeping here

note() { echo "[$(date -u +%H:%M:%S)] $*"; }

# pull_deadline_in EPOCH REASON: move the deadline earlier, never later.
pull_deadline_in() {
  local new=$1 cur
  cur=$(cat "$DEADLINE")
  if [ "$new" -lt "$cur" ]; then
    echo "$new" > "$DEADLINE"
    note "deadline moved in to $(date -u -d "@$new" '+%H:%M') UTC: $2"
  fi
}

# Call progress_start once setup and data prep are done, then `progress i n`
# after each of n units of work (runs, epochs...). From the second unit on, the
# deadline is pulled in to 1.5x the projected finish plus 20 minutes.
progress_start() { date +%s > "$S/progress_start"; }
progress() {
  local done=$1 total=$2 now start proj
  [ "$done" -ge 2 ] || return 0
  now=$(date +%s)
  start=$(cat "$S/progress_start" 2>/dev/null || cat "$S/boot_time")
  proj=$(( (now - start) * total / done ))
  note "progress $done/$total, projected finish $(date -u -d "@$((start + proj))" '+%H:%M') UTC"
  pull_deadline_in $(( start + proj * 3 / 2 + 1200 )) "projected from progress"
}
