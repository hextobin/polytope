#!/bin/bash
# Push /root/logs and everything in out/ to the `results` branch. Files over
# 95MB are skipped (GitHub's limit is 100MB) and listed in logs/skipped.txt.
# Runs every 15 minutes and once more before self-destruct.
set -uo pipefail
source /root/.exp.env
source /root/exp/box/lib.sh
R=$S/results
URL="https://x-access-token:${GH_TOKEN}@github.com/${REPO}.git"

if [ ! -d "$R/.git" ]; then
  git clone -q --branch results "$URL" "$R" 2>/dev/null || {
    rm -rf "$R" && mkdir -p "$R"
    git -C "$R" init -q -b results && git -C "$R" remote add origin "$URL"
  }
fi

mkdir -p "$R/logs" "$R/out"
cp -f "$LOGS"/* "$R/logs/" 2>/dev/null
if [ -d "$OUT" ]; then
  (cd "$OUT" && find . -type f -size -95M -print0 | while IFS= read -r -d '' f; do
    mkdir -p "$R/out/$(dirname "$f")" && cp -fp "$f" "$R/out/$f"
  done)
  (cd "$OUT" && find . -type f -size +95M) > "$R/logs/skipped.txt"
fi

cd "$R"
git add -A
git -c user.name=exp-box -c user.email=box@exp.invalid commit -qm "results $(date -u +%FT%TZ)" || true
git push -q origin results
