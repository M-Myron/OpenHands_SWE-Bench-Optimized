#!/usr/bin/env bash
# run_infer_then_eval.sh
#
# Run inference and evaluation INTERLEAVED per-instance using ThinDockerRuntime.
# For each instance, the same worker:
#   1. pulls the image (cached for the rest of the run),
#   2. runs the agent (inference),
#   3. immediately evaluates the produced patch in a fresh container against
#      the *already-local* image, then moves to the next instance.
#
# Why: the legacy two-phase pipeline (`run_infer` then `eval_infer`) starts
# the eval phase from scratch with all workers cold, causing a burst of
# parallel pulls on overlapping base layers and triggering containerd
# concurrent-pull races (`commit failed: rename ... no such file`,
# `failed to extract layer ... link ...`). Interleaving keeps every image
# warm at the moment eval needs it.
#
# Outputs (same paths/format as the legacy pipeline; downstream tooling works):
#   <out>/output.jsonl                 - inference rows
#   <out>/output.swebench_eval.jsonl   - eval rows
#   <out>/output.swebench_eval.logs/   - per-instance eval logs
#
# Usage:
#   ./run_infer_then_eval.sh <MODEL_CONFIG> <COMMIT_HASH> <AGENT> <EVAL_LIMIT> \
#       <MAX_ITER> <NUM_WORKERS> <DATASET> <SPLIT> [MODE] [EVAL_OUTPUT_DIR] \
#       [PREPARE_ENV_MAX_WORKERS]
# ./evaluation/benchmarks/swe_bench_thin/scripts/run_infer_then_eval.sh llm.eval_policy_traj_128k_swegym_resolved_634i_qwen2-5_coder_32b_full_128k_megatron HEAD CodeActAgent 500 100 16 princeton-nlp/SWE-bench_Verified test

set -eo pipefail

source "evaluation/utils/version_control.sh"

MODEL_CONFIG=${1:-"llm.eval_glm5_fp8_t0"}
COMMIT_HASH=${2:-"HEAD"}
AGENT=${3:-"CodeActAgent"}
EVAL_LIMIT=${4:-"500"}
MAX_ITER=${5:-"100"}
NUM_WORKERS=${6:-"4"}
DATASET=${7:-"princeton-nlp/SWE-bench_Verified"}
SPLIT=${8:-"test"}
MODE=${9:-"swe"}
EVAL_OUTPUT_DIR=${10:-${EVAL_OUTPUT_DIR:-"evaluation/evaluation_outputs/outputs_thin"}}
PREPARE_ENV_MAX_WORKERS=${11:-${OH_RUNTIME_PREPARE_MAX_CONCURRENCY:-8}}

if [ -n "$PREPARE_ENV_MAX_WORKERS" ] && [ "$PREPARE_ENV_MAX_WORKERS" -gt 0 ] 2>/dev/null; then
    export OH_RUNTIME_PREPARE_MAX_CONCURRENCY=$PREPARE_ENV_MAX_WORKERS
fi

checkout_eval_branch

# ── Thin-docker runtime config (mirrors run_infer.sh) ── #
export RUNTIME=thin_docker
export RUN_WITH_BROWSING=false
export DEFAULT_RUNTIME_RESOURCE_FACTOR=2
export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=true

# Stuck-relaunch guards
export OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS=${OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS:-30}
export OH_RUNTIME_PREPARE_TIMEOUT_SECONDS=${OH_RUNTIME_PREPARE_TIMEOUT_SECONDS:-0}
export OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS=${OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS:-7200}

# Larger Docker client timeouts to weather concurrent pull/cp pressure
export DOCKER_CLIENT_TIMEOUT=${DOCKER_CLIENT_TIMEOUT:-300}
export COMPOSE_HTTP_TIMEOUT=${COMPOSE_HTTP_TIMEOUT:-300}

# Image cleanup after each instance finishes inline eval (prevents disk full).
# Set to "false" to retain images for faster re-runs.
export EVAL_CLEANUP_IMAGES=${EVAL_CLEANUP_IMAGES:-true}

get_openhands_version

if [ -z "$USE_HINT_TEXT" ]; then
    export USE_HINT_TEXT=false
fi

EVAL_NOTE="$OPENHANDS_VERSION-thin-interleaved"
[ "$USE_HINT_TEXT" = false ] && EVAL_NOTE="$EVAL_NOTE-no-hint"
[ -n "$EXP_NAME" ] && EVAL_NOTE="$EVAL_NOTE-$EXP_NAME"
[ "$MODE" != "swe" ] && EVAL_NOTE="${EVAL_NOTE}-${MODE}"

echo "================================================================"
echo "SWE-bench Thin Docker  — INTERLEAVED  infer → eval (per-instance)"
echo "================================================================"
echo "AGENT:                  $AGENT"
echo "OPENHANDS_VERSION:      $OPENHANDS_VERSION"
echo "MODEL_CONFIG:           $MODEL_CONFIG"
echo "DATASET:                $DATASET"
echo "SPLIT:                  $SPLIT"
echo "MODE:                   $MODE"
echo "MAX_ITER:               $MAX_ITER"
echo "NUM_WORKERS:            $NUM_WORKERS"
echo "EVAL_LIMIT:             $EVAL_LIMIT"
echo "USE_HINT_TEXT:          $USE_HINT_TEXT"
echo "EVAL_OUTPUT_DIR:        $EVAL_OUTPUT_DIR"
echo "EVAL_NOTE:              $EVAL_NOTE"
if [ -n "$OH_RUNTIME_PREPARE_MAX_CONCURRENCY" ] && [ "$OH_RUNTIME_PREPARE_MAX_CONCURRENCY" -gt 0 ] 2>/dev/null; then
    echo "OH_RUNTIME_PREPARE_MAX_CONCURRENCY: $OH_RUNTIME_PREPARE_MAX_CONCURRENCY"
else
    echo "OH_RUNTIME_PREPARE_MAX_CONCURRENCY: disabled"
fi
echo "================================================================"

unset SANDBOX_ENV_GITHUB_TOKEN

COMMAND="poetry run python evaluation/benchmarks/swe_bench_thin/run_infer_then_eval.py \
    --agent-cls $AGENT \
    --llm-config $MODEL_CONFIG \
    --max-iterations $MAX_ITER \
    --eval-num-workers $NUM_WORKERS \
    --eval-note $EVAL_NOTE \
    --eval-output-dir $EVAL_OUTPUT_DIR \
    --dataset $DATASET \
    --split $SPLIT \
    --mode $MODE"

[ -n "$EVAL_LIMIT" ] && COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"

eval $COMMAND

checkout_original_branch
