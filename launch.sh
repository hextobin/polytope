#!/bin/bash
# Run on your Mac. Creates a GPU droplet that runs box/experiment.sh, pushes
# results to this repo's `results` branch, and destroys itself. Tokens are typed
# at the prompt and only go into the droplet's startup script.
#
#   ./launch.sh --check [deadline_hours]   list the options and prices; rents nothing
#   ./launch.sh [deadline_hours]           (default: DEADLINE_HOURS in experiment.conf)
#
# Nothing is rented without a "y": before each create, launch.sh shows the size,
# region, hourly price and the cost if the box runs to its deadline.
set -euo pipefail
cd "$(dirname "$0")"
source experiment.conf
CHECK=0
if [ "${1:-}" = "--check" ]; then CHECK=1; shift; fi
HOURS=${1:-$DEADLINE_HOURS}
REPO=$(git remote get-url origin | sed -E 's#^(git@github\.com:|https://github\.com/)##; s#\.git$##')
NAME=$(basename "$REPO")-run

# slug, hourly price, comma-separated regions where the size is offered
SIZES_JSON=$(doctl compute size list -o json) || {
  echo "doctl compute size list failed:" >&2; echo "$SIZES_JSON" >&2; exit 1; }
OFFERED=$(jq -r '.[] | "\(.slug) \(.price_hourly) ,\((.regions // []) | join(",")),"' <<<"$SIZES_JSON")

# Options in launch order: "size region price max_cost" for each offered pair.
OPTIONS=()
echo "Options, in launch order (max = hourly price x $HOURS h deadline):"
for size in $SIZES; do
  read -r _ price regions <<<"$(grep "^$size " <<<"$OFFERED" || true)"
  [ -n "${price:-}" ] || { echo "  $size: not offered"; continue; }
  for region in $REGIONS; do
    [[ $regions == *",$region,"* ]] || continue
    max=$(awk -v p="$price" -v h="$HOURS" 'BEGIN { printf "%.2f", p * h }')
    price=$(awk -v p="$price" 'BEGIN { printf "%.2f", p }')
    OPTIONS+=("$size $region $price $max")
    printf '  %d. %-22s %-5s $%s/hr  max $%s\n' "${#OPTIONS[@]}" "$size" "$region" "$price" "$max"
  done
done
if [ ${#OPTIONS[@]} -eq 0 ]; then
  echo "No size in SIZES is offered in REGIONS. Nothing was created." >&2
  exit 1
fi
echo "Capacity is only known when creating; launch.sh asks before each option it tries."
[ "$CHECK" -eq 0 ] || exit 0

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

for opt in "${OPTIONS[@]}"; do
  read -r size region price max <<<"$opt"
  read -rp "Rent $size in $region at \$$price/hr (max \$$max for $HOURS h)? [y = rent, n = next option, Enter = quit] " ans
  case $ans in
    y|Y) ;;
    n|N) continue ;;
    *) echo "Nothing was created."; exit 0 ;;
  esac
  echo "Creating $size in $region. GPU droplets take 5-10 min to come up..."
  if out=$(doctl compute droplet create "$NAME" \
      --size "$size" --region "$region" --image "$IMAGE" \
      --ssh-keys "$SSH_KEY_ID" --user-data-file "$UD" --wait \
      --format ID,Name,PublicIPv4,Status 2>&1); then
    echo "$out"
    echo "Launched: $size in $region at \$$price/hr. Hard deadline: $HOURS h (max \$$max)."
    echo "Results go to the results branch every 15 min."
    exit 0
  fi
  if grep -qi "not available" <<<"$out"; then echo "  no capacity"; continue; fi
  echo "$out" >&2
  exit 1   # any other error: stop rather than risk a second droplet
done
echo "No option was rented. Nothing was created." >&2
exit 1
