#!/usr/bin/env bash
set -Eeuo pipefail


# chmod +x docker_cleanup_old.sh

# # preview only
# DRY_RUN=1 AGE_HOURS=5 ./docker_cleanup_old.sh

# # actually execute
# DRY_RUN=0 AGE_HOURS=5 ./docker_cleanup_old.sh

# # run periodically (every 1.5 hours, clean containers older than 5h)
# DRY_RUN=0 AGE_HOURS=5 LOOP_HOURS=1.5 ./docker_cleanup_old.sh


# ---- config ----
AGE_HOURS="${AGE_HOURS:-5}"
DRY_RUN="${DRY_RUN:-1}"           # 1 = print only, 0 = execute
LOOP_HOURS="${LOOP_HOURS:-0}"     # 0 = run once, >0 = hours between runs (supports decimals)

# Match the container sets from your paste.
NAME_REGEX='^(openhands-runtime-|sweb\.eval\.)'
IMAGE_REGEX='^(mmr1115/openhands-runtime|swebench/)'

# GNU date is assumed (Linux). On macOS, install coreutils and replace `date` with `gdate`.
DATE_BIN="${DATE_BIN:-date}"

cleanup_once() {

cutoff_epoch="$($DATE_BIN -d "${AGE_HOURS} hours ago" +%s)"

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" == "0" ]]; then
    eval "$@"
  fi
}

echo "Cutoff: containers started before $($DATE_BIN -d "@$cutoff_epoch" --iso-8601=seconds)"
echo "Mode: $([[ "$DRY_RUN" == "1" ]] && echo DRY-RUN || echo EXECUTE)"
echo

matched=0

while IFS= read -r cid; do
  [[ -z "$cid" ]] && continue

  # id|name|image|running|startedAt
  line="$(docker inspect --format '{{.Id}}|{{.Name}}|{{.Config.Image}}|{{.State.Running}}|{{.State.StartedAt}}' "$cid" 2>/dev/null || true)"
  [[ -z "$line" ]] && continue

  IFS='|' read -r full_id raw_name image running started_at <<< "$line"
  name="${raw_name#/}"

  # Only target the families from your pasted list
  if [[ ! "$name" =~ $NAME_REGEX && ! "$image" =~ $IMAGE_REGEX ]]; then
    continue
  fi

  # Skip never-started / invalid timestamps
  if [[ "$started_at" == "0001-01-01T00:00:00Z" || -z "$started_at" ]]; then
    continue
  fi

  started_epoch="$($DATE_BIN -d "$started_at" +%s 2>/dev/null || echo 0)"
  [[ "$started_epoch" == "0" ]] && continue

  # Only clean containers started more than AGE_HOURS ago
  if (( started_epoch > cutoff_epoch )); then
    continue
  fi

  matched=1
  echo "MATCH  id=${full_id:0:12}  name=$name  image=$image  started_at=$started_at  running=$running"

  if [[ "$running" == "true" ]]; then
    run "docker stop -t 10 '$full_id'"
  fi

  # remove container after it is stopped
  run "docker rm -f '$full_id'"
  echo
done < <(docker ps -aq --no-trunc)

if [[ "$matched" == "0" ]]; then
  echo "No matching containers older than ${AGE_HOURS}h were found."
  echo
fi

# Catch any other stopped containers older than cutoff
run "docker container prune -f --filter 'until=${AGE_HOURS}h'"

# Remove unused images older than cutoff
run "docker image prune -a -f --filter 'until=${AGE_HOURS}h'"

# Remove old build cache
run "docker builder prune -a -f --filter 'until=${AGE_HOURS}h'"

# Also prune buildx cache when buildx is available
if docker buildx version >/dev/null 2>&1; then
  run "docker buildx prune -a -f --filter 'until=${AGE_HOURS}h'"
fi

echo
echo "Done."
echo "Run with DRY_RUN=0 to actually execute."

} # end cleanup_once

# ---- main ----
LOOP_SECONDS=$(awk "BEGIN {printf \"%d\", ${LOOP_HOURS} * 3600}")
if (( LOOP_SECONDS > 0 )); then
  echo "=== Periodic mode: running every ${LOOP_HOURS}h (${LOOP_SECONDS}s) ==="
  while true; do
    echo ""
    echo "=== Cleanup run at $($DATE_BIN --iso-8601=seconds) ==="
    cleanup_once
    echo "=== Sleeping ${LOOP_HOURS}h until next run ==="
    sleep "$LOOP_SECONDS"
  done
else
  cleanup_once
fi