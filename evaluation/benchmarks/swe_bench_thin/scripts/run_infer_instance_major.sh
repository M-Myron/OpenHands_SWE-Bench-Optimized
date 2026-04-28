#!/usr/bin/env bash
set -eo pipefail

# Instance-major inference for SWE-bench using ThinDockerRuntime.
# Wraps swe_bench_optimized/run_infer_instance_major.py with RUNTIME=thin_docker.
#
# Usage:
#   ./run_infer_instance_major.sh <MODEL_CONFIG> <COMMIT_HASH> <AGENT> <EVAL_LIMIT> \
#       <MAX_ITER> <NUM_WORKERS> <DATASET> <SPLIT> <N_RUNS> <MODE> \
#       [EVAL_OUTPUT_DIR]

source "evaluation/utils/version_control.sh"

MODEL_CONFIG=${1:-"llm.eval_glm5_fp8_t0"}
COMMIT_HASH=${2:-"HEAD"}
AGENT=${3:-"CodeActAgent"}
EVAL_LIMIT=${4:-"500"}
MAX_ITER=${5:-"100"}
NUM_WORKERS=${6:-"4"}
DATASET=${7:-"princeton-nlp/SWE-bench_Verified"}
SPLIT=${8:-"test"}
N_RUNS=${9:-1}
MODE=${10:-"swe"}
EVAL_OUTPUT_DIR=${11:-${EVAL_OUTPUT_DIR:-"evaluation/evaluation_outputs/outputs_thin"}}
PREPARE_ENV_MAX_WORKERS=${12:-${OH_RUNTIME_PREPARE_MAX_CONCURRENCY:-8}}

if [ -n "$PREPARE_ENV_MAX_WORKERS" ] && [ "$PREPARE_ENV_MAX_WORKERS" -gt 0 ] 2>/dev/null; then
    export OH_RUNTIME_PREPARE_MAX_CONCURRENCY=$PREPARE_ENV_MAX_WORKERS
fi

checkout_eval_branch

if [ -z "$AGENT" ]; then AGENT="CodeActAgent"; fi
if [ -z "$MAX_ITER" ]; then MAX_ITER=100; fi
if [ -z "$DATASET" ]; then DATASET="princeton-nlp/SWE-bench_Verified"; fi
if [ -z "$SPLIT" ]; then SPLIT="test"; fi
if [ -z "$MODE" ]; then MODE="swe"; fi
if [ -z "$N_RUNS" ]; then N_RUNS=1; fi

# Force thin_docker runtime
export RUNTIME=thin_docker
export RUN_WITH_BROWSING=false
export DEFAULT_RUNTIME_RESOURCE_FACTOR=2
export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=true

# Guard against stuck relaunches caused by stale env-prepare lock files.
export OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS=${OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS:-30}
export OH_RUNTIME_PREPARE_TIMEOUT_SECONDS=${OH_RUNTIME_PREPARE_TIMEOUT_SECONDS:-0}
export OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS=${OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS:-7200}

get_openhands_version

echo "======================================"
echo "SWE-bench Thin Docker (Instance-Major)"
echo "======================================"
echo "AGENT: $AGENT"
echo "OPENHANDS_VERSION: $OPENHANDS_VERSION"
echo "MODEL_CONFIG: $MODEL_CONFIG"
echo "DATASET: $DATASET"
echo "SPLIT: $SPLIT"
echo "MAX_ITER: $MAX_ITER"
echo "NUM_WORKERS: $NUM_WORKERS"
echo "N_RUNS: $N_RUNS"
echo "MODE: $MODE"
echo "RUNTIME: thin_docker"
echo "EVAL_OUTPUT_DIR: $EVAL_OUTPUT_DIR"
if [ -n "$OH_RUNTIME_PREPARE_MAX_CONCURRENCY" ] && [ "$OH_RUNTIME_PREPARE_MAX_CONCURRENCY" -gt 0 ] 2>/dev/null; then
    echo "OH_RUNTIME_PREPARE_MAX_CONCURRENCY: $OH_RUNTIME_PREPARE_MAX_CONCURRENCY"
else
    echo "OH_RUNTIME_PREPARE_MAX_CONCURRENCY: disabled"
fi
echo "======================================"

if [ -z "$USE_HINT_TEXT" ]; then
    export USE_HINT_TEXT=false
fi

EVAL_NOTE="$OPENHANDS_VERSION-thin"
[ "$USE_HINT_TEXT" = false ] && EVAL_NOTE="$EVAL_NOTE-no-hint"
[ -n "$EXP_NAME" ] && EVAL_NOTE="$EVAL_NOTE-$EXP_NAME"
[ "$MODE" != "swe" ] && EVAL_NOTE="${EVAL_NOTE}-${MODE}"

unset SANDBOX_ENV_GITHUB_TOKEN

COMMAND="poetry run python evaluation/benchmarks/swe_bench_optimized/run_infer_instance_major.py \
    --agent-cls $AGENT \
    --llm-config $MODEL_CONFIG \
    --max-iterations $MAX_ITER \
    --eval-num-workers $NUM_WORKERS \
    --eval-note $EVAL_NOTE \
    --eval-output-dir $EVAL_OUTPUT_DIR \
    --dataset $DATASET \
    --split $SPLIT \
    --mode $MODE \
    --n-runs $N_RUNS"

[ -n "$EVAL_LIMIT" ] && COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"

eval $COMMAND

checkout_original_branch
