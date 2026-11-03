#!/bin/bash
# Final results push, then delete this droplet (and only this one).
# Safe to call more than once: later calls exit at once.
#   self_destruct.sh "reason"
source /root/.exp.env
source /root/exp/box/lib.sh
mkdir "$S/.self_destruct" 2>/dev/null || exit 0
mkdir -p "$LOGS"
echo "[$(date -u +%FT%TZ)] self-destruct: ${1:-no reason given}" >> "$LOGS/self_destruct.log"

# Results first, but never let a stuck push keep the box alive.
for attempt in 1 2 3; do
  timeout 300 "$EXP/box/sync.sh" && break
  sleep 30
done

ID=$(curl -s http://169.254.169.254/metadata/v1/id)
until curl -sf -X DELETE -H "Authorization: Bearer $DO_TOKEN" \
    "https://api.digitalocean.com/v2/droplets/$ID"; do
  echo "[$(date -u +%FT%TZ)] delete failed, retrying in 60s" >> "$LOGS/self_destruct.log"
  sleep 60
done
