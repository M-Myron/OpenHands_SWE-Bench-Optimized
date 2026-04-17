
#!/bin/bash

# NOTE: Instance-major rollout for SWE-Gym TRAINING.
# It keeps per-run output directories while scheduling work in this order:
# instance1_run1..runN, instance2_run1..runN, ...

MODEL=$1
EXP_NAME=$2
N_WORKERS=${3:-64}
N_RUNS=${4:-1}
# Optional: override output directory via 5th arg or EVAL_OUTPUT_DIR env var
EVAL_OUTPUT_DIR=${5:-${EVAL_OUTPUT_DIR:-"evaluation/evaluation_outputs/outputs"}}
# Optional: limit concurrent runtime environment preparation during inference workers.
# 0/empty means no limit (default behavior).
PREPARE_ENV_MAX_WORKERS=${6:-${OH_RUNTIME_PREPARE_MAX_CONCURRENCY:-0}}

if [ -n "$PREPARE_ENV_MAX_WORKERS" ] && [ "$PREPARE_ENV_MAX_WORKERS" -gt 0 ] 2>/dev/null; then
    export OH_RUNTIME_PREPARE_MAX_CONCURRENCY=$PREPARE_ENV_MAX_WORKERS
fi

export EXP_NAME=$EXP_NAME
export DEFAULT_RUNTIME_RESOURCE_FACTOR=2
export ITERATIVE_EVAL_MODE=false
export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=true
# Guard against stuck relaunches caused by stale env-prepare lock files.
export OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS=${OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS:-30}
export OH_RUNTIME_PREPARE_TIMEOUT_SECONDS=${OH_RUNTIME_PREPARE_TIMEOUT_SECONDS:-0}
export OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS=${OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS:-7200}

echo "MODEL: $MODEL"
echo "EXP_NAME: $EXP_NAME"
echo "N_WORKERS: $N_WORKERS"
echo "N_RUNS: $N_RUNS"
if [ -n "$OH_RUNTIME_PREPARE_MAX_CONCURRENCY" ] && [ "$OH_RUNTIME_PREPARE_MAX_CONCURRENCY" -gt 0 ] 2>/dev/null; then
    echo "OH_RUNTIME_PREPARE_MAX_CONCURRENCY: $OH_RUNTIME_PREPARE_MAX_CONCURRENCY"
else
    echo "OH_RUNTIME_PREPARE_MAX_CONCURRENCY: disabled"
fi
echo "OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS: $OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS"
echo "OH_RUNTIME_PREPARE_TIMEOUT_SECONDS: $OH_RUNTIME_PREPARE_TIMEOUT_SECONDS"
echo "OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS: $OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS"

DATASET="SWE-Gym/SWE-Gym"
SPLIT="train"

if [ -z "$ALLHANDS_API_KEY" ]; then
    echo "ALLHANDS_API_KEY is not set. Will rollout and evaluate locally using Docker. WARNING: A large value of N_WORKERS will result in a large number of Docker containers being spun up and may crash your machine."
    export RUNTIME=docker
else
    echo "ALLHANDS_API_KEY is set. Continuing rollout and evaluation with remote runtime..."
    export RUNTIME=remote
    export SANDBOX_REMOTE_RUNTIME_API_URL="https://runtime.eval.all-hands.dev"
    export EVAL_DOCKER_IMAGE_PREFIX="us-central1-docker.pkg.dev/evaluation-092424/swe-bench-images"
fi

EVAL_LIMIT=3000
MAX_ITER=100

source "evaluation/utils/version_control.sh"
get_openhands_version

echo "OPENHANDS_VERSION: $OPENHANDS_VERSION"
echo "MODEL_CONFIG: $MODEL_CONFIG"
echo "DATASET: $DATASET"
echo "SPLIT: $SPLIT"

# Default to NOT use Hint
export USE_INSTANCE_IMAGE=true
export USE_HINT_TEXT=false
export RUN_WITH_BROWSING=false
echo "USE_HINT_TEXT: $USE_HINT_TEXT"
EVAL_NOTE="$OPENHANDS_VERSION-no-hint-$EXP_NAME"

function run_eval() {
  local eval_note=$1
  local n_runs=$2

  COMMAND="poetry run python evaluation/benchmarks/swe_bench_optimized/run_infer_instance_major.py \
    --agent-cls CodeActAgent \
    --llm-config $MODEL \
    --max-iterations $MAX_ITER \
    --eval-num-workers $N_WORKERS \
    --eval-note $eval_note \
    --eval-output-dir $EVAL_OUTPUT_DIR \
    --dataset $DATASET \
    --split $SPLIT \
    --n-runs $n_runs"

  if [ -n "$EVAL_LIMIT" ]; then
    echo "EVAL_LIMIT: $EVAL_LIMIT"
    COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
  fi

  eval $COMMAND
}

echo "### Running inference with N_RUNS=$N_RUNS using instance-major scheduling ###"
unset SANDBOX_ENV_GITHUB_TOKEN

while true; do
    echo "### Running inference... ###"
    INFER_OUTPUT=$(run_eval "$EVAL_NOTE" "$N_RUNS")
    INFER_STATUS=$?
    echo "INFER_STATUS: $INFER_STATUS"

    echo "### Cleaning up remote runtime... ###"
    # ./evaluation/utils/scripts/cleanup_remote_runtime.sh

    if [ "$RUNTIME" = "docker" ]; then
        echo "### Cleaning up local docker containers... ###"
        docker ps -q --filter "name=openhands-runtime-" | xargs -r docker stop
        docker ps -aq --filter "name=openhands-runtime-" | xargs -r docker rm

        echo "### Cleaning up SWE-bench related docker images... ###"
        docker images --format "{{.Repository}}:{{.Tag}}" | grep -E "^(us-central1-docker\.pkg\.dev/evaluation-092424/swe-bench-images|docker\.io/xingyaoww/|mmr1115/openhands-runtime|ghcr\.io/openhands/runtime)" | xargs -r docker rmi -f 2>/dev/null || true

        echo "### Pruning dangling images and build cache... ###"
        docker image prune -f 2>/dev/null || true
        docker builder prune -f --filter "until=1h" 2>/dev/null || true

        echo "### Docker cleanup completed. Current usage: ###"
        docker system df
    fi

    if [ $INFER_STATUS -eq 0 ]; then
        echo "### Inference completed successfully for all runs. ###"
        break
    else
        echo "### Inference failed with exit code $INFER_STATUS. Retrying... ###"
    fi
done

# ============================================================
# Evaluate all runs
# ============================================================
echo "### Starting evaluation for all $N_RUNS runs ###"

OUTPUT_FILES=()
for run_idx in $(seq 1 $N_RUNS); do
    if [ $N_RUNS -gt 1 ]; then
        OUTPUT_FILE=$(echo "$INFER_OUTPUT" | grep -o "### OUTPUT FILE FOR RUN $run_idx:.* ###" | sed "s/### OUTPUT FILE FOR RUN $run_idx: \(.*\) ###/\1/")
    else
        OUTPUT_FILE=$(echo "$INFER_OUTPUT" | grep -o '### OUTPUT FILE:.* ###' | sed 's/### OUTPUT FILE: \(.*\) ###/\1/')
    fi

    if [ -z "$OUTPUT_FILE" ]; then
        echo "Warning: Could not extract OUTPUT_FILE for run $run_idx from inference output"
        BASE_OUTPUT_DIR="evaluation/evaluation_outputs/outputs"
        if [ $N_RUNS -gt 1 ]; then
            PATTERN="${EVAL_NOTE}-run_${run_idx}"
        else
            PATTERN="${EVAL_NOTE}"
        fi
        OUTPUT_FILE=$(find "$BASE_OUTPUT_DIR" -name "output.jsonl" -path "*${PATTERN}*" | head -n 1)
    fi

    if [ -n "$OUTPUT_FILE" ] && [ -f "$OUTPUT_FILE" ]; then
        echo "Found OUTPUT_FILE for run $run_idx: $OUTPUT_FILE"
        OUTPUT_FILES+=("$OUTPUT_FILE")
    else
        echo "Error: OUTPUT_FILE not found for run $run_idx"
    fi
done

for run_idx in $(seq 1 $N_RUNS); do
    OUTPUT_FILE="${OUTPUT_FILES[$((run_idx-1))]}"

    if [ -z "$OUTPUT_FILE" ]; then
        echo "### Skipping evaluation for run $run_idx (output file not found) ###"
        continue
    fi

    echo "### Evaluating run $run_idx/$N_RUNS on $OUTPUT_FILE ... ###"

    while true; do
        COMMAND="poetry run python evaluation/benchmarks/swe_bench/eval_infer.py \
        --eval-num-workers $((N_WORKERS * 2)) \
        --input-file $OUTPUT_FILE \
        --dataset $DATASET \
        --split $SPLIT"

        if [ -n "$EVAL_LIMIT" ]; then
            echo "EVAL_LIMIT: $EVAL_LIMIT"
            COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
        fi
        echo "Running command: $COMMAND"

        eval $COMMAND
        EVAL_STATUS=$?

        if [ $EVAL_STATUS -eq 0 ]; then
            echo "### Evaluation completed successfully for run $run_idx. ###"
            break
        else
            echo "### Evaluation failed with exit code $EVAL_STATUS. Retrying... ###"
        fi

        ./evaluation/utils/scripts/cleanup_remote_runtime.sh

        if [ "$RUNTIME" = "docker" ]; then
            echo "### Cleaning up local docker containers... ###"
            docker ps -q --filter "name=openhands-runtime-" | xargs -r docker stop
            docker ps -aq --filter "name=openhands-runtime-" | xargs -r docker rm

            echo "### Cleaning up SWE-bench related docker images... ###"
            docker images --format "{{.Repository}}:{{.Tag}}" | grep -E "^(us-central1-docker\.pkg\.dev/evaluation-092424/swe-bench-images|docker\.io/xingyaoww/|mmr1115/openhands-runtime|ghcr\.io/openhands/runtime)" | xargs -r docker rmi -f 2>/dev/null || true

            echo "### Pruning dangling images and build cache... ###"
            docker image prune -f 2>/dev/null || true
            docker builder prune -f --filter "until=1h" 2>/dev/null || true

            echo "### Docker cleanup completed. Current usage: ###"
            docker system df
        fi
    done

    echo "### Updating the output with evaluation results for run $run_idx... ###"
    poetry run python evaluation/benchmarks/swe_bench/scripts/eval/update_output_with_eval.py $OUTPUT_FILE

    echo "### Combining the final completions for run $run_idx... ###"
    poetry run python evaluation/benchmarks/swe_bench/scripts/eval/combine_final_completions.py $OUTPUT_FILE

    echo "### DONE for run $run_idx! ###"
    echo "You can find the final output at $(dirname $OUTPUT_FILE)/$FINAL_OUTPUT_FILE"
done

echo ""
echo "=============================================================="
echo "ALL $N_RUNS RUNS COMPLETED!"
echo "=============================================================="
