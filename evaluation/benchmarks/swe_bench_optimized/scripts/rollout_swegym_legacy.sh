#!/bin/bash

# SWE-Gym Legacy Format: Roll out trajectories using the original SWE-Gym paper's format.
#
# This ensures trajectories match the format of the released SFT data:
# https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories
#
# Key differences from rollout_swegym.sh:
# - Uses SweGymLegacyCodeActAgent (3 simple tools, no security_risk, no think, no task_tracker)
# - Uses the original simple system prompt (no ROLE/EFFICIENCY/etc. sections)
# - Uses the original 5-step task instruction format
#
# Usage:
#   bash evaluation/benchmarks/swe_bench_optimized/scripts/rollout_swegym_legacy.sh \
#       llm.mymodel 'train-legacy-t05' 16 1

MODEL=$1
EXP_NAME=$2
N_WORKERS=${3:-64}
N_RUNS=${4:-1}
EVAL_OUTPUT_DIR=${5:-${EVAL_OUTPUT_DIR:-"evaluation/evaluation_outputs/outputs"}}
PREPARE_ENV_MAX_WORKERS=${6:-${OH_RUNTIME_PREPARE_MAX_CONCURRENCY:-0}}

if [ -n "$PREPARE_ENV_MAX_WORKERS" ] && [ "$PREPARE_ENV_MAX_WORKERS" -gt 0 ] 2>/dev/null; then
    export OH_RUNTIME_PREPARE_MAX_CONCURRENCY=$PREPARE_ENV_MAX_WORKERS
fi

export EXP_NAME=$EXP_NAME
export DEFAULT_RUNTIME_RESOURCE_FACTOR=2
export ITERATIVE_EVAL_MODE=false
# Skip instances that reach maximum retries instead of crashing the entire evaluation
# Failed instances will be logged to maximum_retries_exceeded.jsonl
export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=true
export ITERATIVE_EVAL_MODE=false


echo "=== SWE-Gym Legacy Format Rollout ==="
echo "NOTE: Using original SWE-Gym paper's prompt/tool format"
echo "MODEL: $MODEL"
echo "EXP_NAME: $EXP_NAME"
echo "N_WORKERS: $N_WORKERS"
echo "N_RUNS: $N_RUNS"

DATASET="SWE-Gym/SWE-Gym"
SPLIT="train"

if [ -z "$ALLHANDS_API_KEY" ]; then
    echo "Running locally with Docker."
    export RUNTIME=docker
else
    echo "Running with remote runtime."
    export RUNTIME=remote
    export SANDBOX_REMOTE_RUNTIME_API_URL="https://runtime.eval.all-hands.dev"
    export EVAL_DOCKER_IMAGE_PREFIX="us-central1-docker.pkg.dev/evaluation-092424/swe-bench-images"
fi

EVAL_LIMIT=3000
MAX_ITER=100

source "evaluation/utils/version_control.sh"
get_openhands_version

echo "OPENHANDS_VERSION: $OPENHANDS_VERSION"
echo "DATASET: $DATASET"
echo "SPLIT: $SPLIT"

export USE_INSTANCE_IMAGE=true
export USE_HINT_TEXT=false
export RUN_WITH_BROWSING=false

EVAL_NOTE="$OPENHANDS_VERSION-no-hint-swegym-legacy-$EXP_NAME"

function run_eval() {
  local eval_note=$1
  local n_runs=$2

  # Periodic Docker cleanup for local runtime
  if [ "$RUNTIME" = "docker" ]; then
    (
      while true; do
        sleep 1800
        echo "### Running periodic Docker cleanup... ###"
        docker ps -q --filter "name=openhands-runtime-" --filter "status=exited" | xargs -r docker rm 2>/dev/null || true
        docker images --format "{{.Repository}}:{{.Tag}}" | grep -E "^(us-central1-docker\.pkg\.dev/evaluation-092424/swe-bench-images|docker\.io/xingyaoww/|mmr1115/openhands-runtime)" | xargs -r docker rmi -f 2>/dev/null || true
        docker image prune -f 2>/dev/null || true
        docker builder prune -f --filter "until=30m" 2>/dev/null || true
      done
    ) &
    CLEANUP_PID=$!
    trap "kill $CLEANUP_PID 2>/dev/null || true" EXIT INT TERM
  fi

  COMMAND="poetry run python evaluation/benchmarks/swe_bench_optimized/run_infer_swegym_legacy.py \
    --agent-cls SweGymLegacyCodeActAgent \
    --llm-config $MODEL \
    --max-iterations $MAX_ITER \
    --eval-num-workers $N_WORKERS \
    --eval-note $eval_note \
    --eval-output-dir $EVAL_OUTPUT_DIR \
    --dataset $DATASET \
    --split $SPLIT \
    --n-runs $n_runs"

  if [ -n "$EVAL_LIMIT" ]; then
    COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
  fi

  eval $COMMAND

  if [ "$RUNTIME" = "docker" ]; then
    kill $CLEANUP_PID 2>/dev/null || true
    trap - EXIT INT TERM
  fi
}

unset SANDBOX_ENV_GITHUB_TOKEN

echo "### Rolling out with SWE-Gym legacy format, N_RUNS=$N_RUNS ###"

while true; do
    echo "### Running rollout... ###"
    INFER_OUTPUT=$(run_eval "$EVAL_NOTE" "$N_RUNS")
    INFER_STATUS=$?

    if [ "$RUNTIME" = "docker" ]; then
        echo "### Cleaning up Docker containers... ###"
        docker ps -q --filter "name=openhands-runtime-" | xargs -r docker stop
        docker ps -aq --filter "name=openhands-runtime-" | xargs -r docker rm
        docker images --format "{{.Repository}}:{{.Tag}}" | grep -E "^(us-central1-docker\.pkg\.dev/evaluation-092424/swe-bench-images|docker\.io/xingyaoww/|mmr1115/openhands-runtime)" | xargs -r docker rmi -f 2>/dev/null || true
        docker image prune -f 2>/dev/null || true
    fi

    if [ $INFER_STATUS -eq 0 ]; then
        echo "### Rollout completed successfully. ###"
        break
    else
        echo "### Rollout failed with exit code $INFER_STATUS. Retrying... ###"
    fi
done

# Evaluate
echo "### Starting evaluation for all $N_RUNS runs ###"
OUTPUT_FILES=()
for run_idx in $(seq 1 $N_RUNS); do
    if [ $N_RUNS -gt 1 ]; then
        OUTPUT_FILE=$(echo "$INFER_OUTPUT" | grep -o "### OUTPUT FILE FOR RUN $run_idx:.* ###" | sed "s/### OUTPUT FILE FOR RUN $run_idx: \(.*\) ###/\1/")
    else
        OUTPUT_FILE=$(echo "$INFER_OUTPUT" | grep -o '### OUTPUT FILE:.* ###' | sed 's/### OUTPUT FILE: \(.*\) ###/\1/')
    fi

    if [ -z "$OUTPUT_FILE" ]; then
        echo "WARNING: Could not find output file for run $run_idx, skipping evaluation"
        continue
    fi

    OUTPUT_FILES+=("$OUTPUT_FILE")
done

for OUTPUT_FILE in "${OUTPUT_FILES[@]}"; do
    echo "### Evaluating: $OUTPUT_FILE ###"
    bash evaluation/benchmarks/swe_bench_optimized/scripts/eval_infer_swegym_legacy.sh "$OUTPUT_FILE" "" "$DATASET" "$SPLIT" || {
        echo "WARNING: Evaluation failed for $OUTPUT_FILE"
    }
done

echo "### All done! ###"
