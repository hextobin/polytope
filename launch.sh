#!/bin/bash
# Run on your Mac. Creates a GPU droplet that runs box/experiment.sh, pushes
# results to this repo's `results` branch, and destroys itself. Tokens are typed
# at the prompt and only go into the droplet's startup script.
#
#   ./launch.sh [deadline_hours]     (default: DEADLINE_HOURS in experiment.conf)
set -euo pipefail
cd "$(dirname "$0")"
source experiment.conf
HOURS=${1:-$DEADLINE_HOURS}
REPO=$(git remote get-url origin | sed -E 's#^(git@github\.com:|https://github\.com/)##; s#\.git$##')
NAME=$(basename "$REPO")-run

# The box clones origin/main, so local changes must be pushed first.
git fetch -q origin main
if git cat-file -e origin/main:control/stop 2>/dev/null; then
  echo "control/stop is on main (it would destroy the box at once). Remove it first." >&2
  exit 1
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ] || \
   [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
  echo "Local main differs from origin/main. Commit and push first." >&2
  exit 1
fi
if doctl compute droplet list --format Name --no-header | grep -qx "$NAME"; then
  echo "A droplet named $NAME already exists. Not launching a second one." >&2
  exit 1
fi

read -rsp "DigitalOcean token (droplet read + delete): " DO_TOKEN; echo
read -rsp "GitHub token ($REPO contents read/write): " GH_TOKEN; echo

UD=$(mktemp)
trap 'rm -f "$UD"' EXIT
chmod 600 "$UD"
cat > "$UD" <<UDEOF
#!/bin/bash
set -euo pipefail
umask 077
cat > /root/.exp.env <<'ENV'
DO_TOKEN=$DO_TOKEN
GH_TOKEN=$GH_TOKEN
REPO=$REPO
DEADLINE_HOURS=$HOURS
KEEP_AFTER_DONE_MIN=$KEEP_AFTER_DONE_MIN
PIP_PACKAGES="$PIP_PACKAGES"
ENV
git clone -q --single-branch --branch main https://x-access-token:$GH_TOKEN@github.com/$REPO.git /root/exp
bash /root/exp/box/setup.sh
UDEOF

# slug, hourly price, comma-separated regions where the size is offered
OFFERED=$(doctl compute size list -o json |
  jq -r '.[] | "\(.slug) \(.price_hourly) ,\((.regions // []) | join(",")),"')

for size in $SIZES; do
  read -r _ price regions <<<"$(grep "^$size " <<<"$OFFERED" || true)"
  [ -n "${price:-}" ] || { echo "$size: not offered"; continue; }
  for region in $REGIONS; do
    [[ $regions == *",$region,"* ]] || continue
    echo "Trying $size in $region (\$$price/hr). GPU droplets take 5-10 min to come up..."
    if out=$(doctl compute droplet create "$NAME" \
        --size "$size" --region "$region" --image "$IMAGE" \
        --ssh-keys "$SSH_KEY_ID" --user-data-file "$UD" --wait \
        --format ID,Name,PublicIPv4,Status 2>&1); then
      echo "$out"
      echo "Launched. Hard deadline: $HOURS h. Results go to the results branch every 15 min."
      exit 0
    fi
    if grep -qi "not available" <<<"$out"; then echo "  no capacity"; continue; fi
    echo "$out" >&2
    exit 1   # any other error: stop rather than risk a second droplet
  done
done
echo "No capacity for any size in SIZES and REGIONS. Nothing was created." >&2
exit 1
