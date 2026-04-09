#!/usr/bin/env bash
set -eo pipefail

# Instance-major variant of run_infer_swegym_legacy.sh
#
# Uses Pool-based scheduling (no batch boundaries) with SweGymLegacyCodeActAgent.
# Workers pick up the next instance immediately when done — no idle waiting.
# Docker cleanup is handled per-instance inside the Python script.
#
# Usage (same arg order as run_infer_swegym_legacy.sh):
#   bash evaluation/benchmarks/swe_bench_optimized/scripts/run_infer_swegym_legacy_instance_major.sh \
#       MODEL_CONFIG COMMIT_HASH AGENT EVAL_LIMIT MAX_ITER NUM_WORKERS DATASET SPLIT N_RUNS MODE
#
# Examples:
#   # Evaluate on SWE-bench Verified
#   bash evaluation/benchmarks/swe_bench_optimized/scripts/run_infer_swegym_legacy_instance_major.sh \
#       llm.eval_swegym_openhands_sft_qwen2.5_coder_14b_full_32k_torchtune HEAD SweGymLegacyCodeActAgent 500 100 16
#
#   # Evaluate on SWE-bench Lite
#   bash evaluation/benchmarks/swe_bench_optimized/scripts/run_infer_swegym_legacy_instance_major.sh \
#       llm.mymodel HEAD SweGymLegacyCodeActAgent 300 100 16 princeton-nlp/SWE-bench_Lite test
#
#   # Rollout on SWE-Gym (for training data collection)
#   bash evaluation/benchmarks/swe_bench_optimized/scripts/run_infer_swegym_legacy_instance_major.sh \
#       llm.mymodel HEAD SweGymLegacyCodeActAgent 3000 100 64 SWE-Gym/SWE-Gym train
#
#   bash evaluation/benchmarks/swe_bench_optimized/scripts/run_infer_swegym_legacy_instance_major.sh llm.eval_swegym_openhands_sft_qwen2-5_coder_14b_full_32k_torchtune HEAD SweGymLegacyCodeActAgent 500 100 32
#

source "evaluation/utils/version_control.sh"

MODEL_CONFIG=$1
COMMIT_HASH=$2
AGENT=$3
EVAL_LIMIT=$4
MAX_ITER=$5
NUM_WORKERS=$6
DATASET=$7
SPLIT=$8
N_RUNS=$9
MODE=${10}
# Optional: override output directory via 11th arg or EVAL_OUTPUT_DIR env var
EVAL_OUTPUT_DIR=${11:-${EVAL_OUTPUT_DIR:-"evaluation/evaluation_outputs/outputs"}}
# Optional: limit concurrent runtime environment preparation during inference workers.
# 0/empty means no limit (default behavior).
PREPARE_ENV_MAX_WORKERS=${12:-${OH_RUNTIME_PREPARE_MAX_CONCURRENCY:-8}}
# Skip instances that reach maximum retries instead of crashing the entire evaluation
# Failed instances will be logged to maximum_retries_exceeded.jsonl
export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=true

if [ -n "$PREPARE_ENV_MAX_WORKERS" ] && [ "$PREPARE_ENV_MAX_WORKERS" -gt 0 ] 2>/dev/null; then
    export OH_RUNTIME_PREPARE_MAX_CONCURRENCY=$PREPARE_ENV_MAX_WORKERS
fi

if [ -z "$NUM_WORKERS" ]; then
  NUM_WORKERS=1
  echo "Number of workers not specified, use default $NUM_WORKERS"
fi
checkout_eval_branch

if [ -z "$AGENT" ]; then
  echo "Agent not specified, use default SweGymLegacyCodeActAgent"
  AGENT="SweGymLegacyCodeActAgent"
fi

if [ -z "$MAX_ITER" ]; then
  echo "MAX_ITER not specified, use default 100"
  MAX_ITER=100
fi

# Force disable browsing for legacy mode
if [ -z "$RUN_WITH_BROWSING" ]; then
  RUN_WITH_BROWSING=false
fi

if [ -z "$DATASET" ]; then
  echo "DATASET not specified, use default princeton-nlp/SWE-bench_Verified"
  DATASET="princeton-nlp/SWE-bench_Verified"
fi

if [ -z "$SPLIT" ]; then
  echo "SPLIT not specified, use default test"
  SPLIT="test"
fi

if [ -z "$MODE" ]; then
  MODE="swe"
  echo "MODE not specified, use default $MODE"
fi

if [ -n "$EVAL_CONDENSER" ]; then
  echo "Using Condenser Config: $EVAL_CONDENSER"
else
  echo "No Condenser Config provided via EVAL_CONDENSER, use default (NoOpCondenser)."
fi

export RUN_WITH_BROWSING=$RUN_WITH_BROWSING
export DEFAULT_RUNTIME_RESOURCE_FACTOR=2
echo "RUN_WITH_BROWSING: $RUN_WITH_BROWSING"
echo "=== SWE-Gym Legacy Format + Instance-Major Scheduling ==="

# Guard against stuck relaunches caused by stale env-prepare lock files.
export OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS=${OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS:-30}
export OH_RUNTIME_PREPARE_TIMEOUT_SECONDS=${OH_RUNTIME_PREPARE_TIMEOUT_SECONDS:-1800}
export OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS=${OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS:-7200}

get_openhands_version

echo "AGENT: $AGENT"
echo "OPENHANDS_VERSION: $OPENHANDS_VERSION"
echo "MODEL_CONFIG: $MODEL_CONFIG"
echo "DATASET: $DATASET"
echo "SPLIT: $SPLIT"
echo "MAX_ITER: $MAX_ITER"
echo "NUM_WORKERS: $NUM_WORKERS"
echo "COMMIT_HASH: $COMMIT_HASH"
echo "MODE: $MODE"
echo "EVAL_CONDENSER: $EVAL_CONDENSER"
echo "EVAL_OUTPUT_DIR: $EVAL_OUTPUT_DIR"
if [ -n "$OH_RUNTIME_PREPARE_MAX_CONCURRENCY" ] && [ "$OH_RUNTIME_PREPARE_MAX_CONCURRENCY" -gt 0 ] 2>/dev/null; then
    echo "OH_RUNTIME_PREPARE_MAX_CONCURRENCY: $OH_RUNTIME_PREPARE_MAX_CONCURRENCY"
else
    echo "OH_RUNTIME_PREPARE_MAX_CONCURRENCY: disabled"
fi

# Default to NOT use Hint
if [ -z "$USE_HINT_TEXT" ]; then
  export USE_HINT_TEXT=false
fi
echo "USE_HINT_TEXT: $USE_HINT_TEXT"
EVAL_NOTE="$OPENHANDS_VERSION"
# if not using Hint, add -no-hint to the eval note
if [ "$USE_HINT_TEXT" = false ]; then
  EVAL_NOTE="$EVAL_NOTE-no-hint"
fi

if [ "$RUN_WITH_BROWSING" = true ]; then
  EVAL_NOTE="$EVAL_NOTE-with-browsing"
fi

if [ -n "$EXP_NAME" ]; then
  EVAL_NOTE="$EVAL_NOTE-$EXP_NAME"
fi
# if mode != swe, add mode to the eval note
if [ "$MODE" != "swe" ]; then
  EVAL_NOTE="${EVAL_NOTE}-${MODE}"
fi
# Add condenser config to eval note if provided
if [ -n "$EVAL_CONDENSER" ]; then
  EVAL_NOTE="${EVAL_NOTE}-${EVAL_CONDENSER}"
fi

function run_eval() {
  local eval_note="${1}"
  local n_runs="${2}"

  # No background Docker cleanup needed — the instance-major Python script
  # cleans up per-instance after all runs for that instance complete.

  COMMAND="poetry run python evaluation/benchmarks/swe_bench_optimized/run_infer_swegym_legacy_instance_major.py \
    --agent-cls $AGENT \
    --llm-config $MODEL_CONFIG \
    --max-iterations $MAX_ITER \
    --eval-num-workers $NUM_WORKERS \
    --eval-note $eval_note \
    --eval-output-dir $EVAL_OUTPUT_DIR \
    --dataset $DATASET \
    --split $SPLIT \
    --mode $MODE \
    --n-runs $n_runs"

  if [ -n "$EVAL_LIMIT" ]; then
    echo "EVAL_LIMIT: $EVAL_LIMIT"
    COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
  fi

  # Run the command
  eval $COMMAND
}

unset SANDBOX_ENV_GITHUB_TOKEN # prevent the agent from using the github token to push
if [ -z "$N_RUNS" ]; then
  N_RUNS=1
  echo "N_RUNS not specified, use default $N_RUNS"
fi

echo "### Running inference with N_RUNS=$N_RUNS using instance-major scheduling ###"
echo "### SweGym Legacy format (original paper prompt/tool) ###"
echo "### (each instance completes all runs before moving to the next) ###"
run_eval "$EVAL_NOTE" "$N_RUNS"

checkout_original_branch
