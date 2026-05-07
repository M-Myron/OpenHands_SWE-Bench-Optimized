#!/bin/bash

# SWE-Gym TRAINING rollout using ThinDockerRuntime.
# Instance-major scheduling: instance1_run1..runN, instance2_run1..runN, ...
# Wraps the same Python scripts as swe_bench_optimized but with RUNTIME=thin_docker.
# bash evaluation/benchmarks/swe_bench_thin/scripts/rollout_swegym_instance_major.sh llm.eval_glm5_fp8_t05 glm5_fp8_t05_thinking 32 1


MODEL=$1
EXP_NAME=$2
N_WORKERS=${3:-64}
N_RUNS=${4:-1}
EVAL_OUTPUT_DIR=${5:-${EVAL_OUTPUT_DIR:-"evaluation/evaluation_outputs/outputs_thin"}}
PREPARE_ENV_MAX_WORKERS=${6:-${OH_RUNTIME_PREPARE_MAX_CONCURRENCY:-0}}
# AGENT_CONFIG=${AGENT_CONFIG:-"glm5_thinking"}

if [ -n "$PREPARE_ENV_MAX_WORKERS" ] && [ "$PREPARE_ENV_MAX_WORKERS" -gt 0 ] 2>/dev/null; then
    export OH_RUNTIME_PREPARE_MAX_CONCURRENCY=$PREPARE_ENV_MAX_WORKERS
fi

# Force thin_docker runtime
export RUNTIME=thin_docker
export EXP_NAME=$EXP_NAME
export DEFAULT_RUNTIME_RESOURCE_FACTOR=2
export ITERATIVE_EVAL_MODE=false
export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=true
export RUN_WITH_BROWSING=false

# Guard against stuck relaunches
export OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS=${OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS:-30}
export OH_RUNTIME_PREPARE_TIMEOUT_SECONDS=${OH_RUNTIME_PREPARE_TIMEOUT_SECONDS:-0}
export OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS=${OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS:-7200}
export OH_RUNTIME_PREPARE_MAX_CONCURRENCY=${OH_RUNTIME_PREPARE_MAX_CONCURRENCY:-8}

# Remove Docker images after each eval instance completes (prevents disk full).
# Set to "false" to keep images cached for faster re-runs.
# export EVAL_CLEANUP_IMAGES=false
export EVAL_CLEANUP_IMAGES=${EVAL_CLEANUP_IMAGES:-true}

echo "======================================"
echo "SWE-Gym Rollout (Thin Docker)"
echo "======================================"
echo "MODEL: $MODEL"
echo "EXP_NAME: $EXP_NAME"
echo "N_WORKERS: $N_WORKERS"
echo "N_RUNS: $N_RUNS"
echo "RUNTIME: thin_docker"
echo "EVAL_OUTPUT_DIR: $EVAL_OUTPUT_DIR"
if [ -n "$OH_RUNTIME_PREPARE_MAX_CONCURRENCY" ] && [ "$OH_RUNTIME_PREPARE_MAX_CONCURRENCY" -gt 0 ] 2>/dev/null; then
    echo "OH_RUNTIME_PREPARE_MAX_CONCURRENCY: $OH_RUNTIME_PREPARE_MAX_CONCURRENCY"
else
    echo "OH_RUNTIME_PREPARE_MAX_CONCURRENCY: disabled"
fi
echo "======================================"

DATASET="SWE-Gym/SWE-Gym"
SPLIT="train"
EVAL_LIMIT=3000
MAX_ITER=100

source "evaluation/utils/version_control.sh"
get_openhands_version

echo "OPENHANDS_VERSION: $OPENHANDS_VERSION"
echo "DATASET: $DATASET"
echo "SPLIT: $SPLIT"

export USE_INSTANCE_IMAGE=true
export USE_HINT_TEXT=false
EVAL_NOTE="$OPENHANDS_VERSION-thin-no-hint-$EXP_NAME"

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

  # Pass --agent-config if AGENT_CONFIG is set (e.g. glm5_thinking for reasoning models)
  if [ -n "$AGENT_CONFIG" ]; then
    COMMAND="$COMMAND --agent-config $AGENT_CONFIG"
  fi

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

    # Thin docker: only stop/rm containers, skip heavy image cleanup
    echo "### Cleaning up thin docker containers... ###"
    docker ps -q --filter "name=openhands-thin-" | xargs -r docker stop 2>/dev/null || true
    docker ps -aq --filter "name=openhands-thin-" | xargs -r docker rm 2>/dev/null || true

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
        if [ $N_RUNS -gt 1 ]; then
            PATTERN="${EVAL_NOTE}-run_${run_idx}"
        else
            PATTERN="${EVAL_NOTE}"
        fi
        OUTPUT_FILE=$(find "$EVAL_OUTPUT_DIR" -name "output.jsonl" -path "*${PATTERN}*" | head -n 1)
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
        # Ensure thin_docker runtime for eval (avoid full DockerRuntime overlay builds)
        export RUNTIME=thin_docker
        COMMAND="poetry run python evaluation/benchmarks/swe_bench/eval_infer.py \
        --eval-num-workers $((N_WORKERS)) \
        --input-file $OUTPUT_FILE \
        --dataset $DATASET \
        --split $SPLIT"

        if [ -n "$EVAL_LIMIT" ]; then
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

        # Thin cleanup between retries
        docker ps -q --filter "name=openhands-thin-" | xargs -r docker stop 2>/dev/null || true
        docker ps -aq --filter "name=openhands-thin-" | xargs -r docker rm 2>/dev/null || true
    done

    echo "### Updating the output with evaluation results for run $run_idx... ###"
    yes y | poetry run python evaluation/benchmarks/swe_bench/scripts/eval/update_output_with_eval.py $OUTPUT_FILE

    echo "### Combining the final completions for run $run_idx... ###"
    yes y | poetry run python evaluation/benchmarks/swe_bench/scripts/eval/combine_final_completions.py $OUTPUT_FILE

    echo "### DONE for run $run_idx! ###"
done

echo ""
echo "=============================================================="
echo "ALL $N_RUNS RUNS COMPLETED! (thin docker)"
echo "=============================================================="
