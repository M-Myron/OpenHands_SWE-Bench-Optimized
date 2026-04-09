#!/usr/bin/env bash
set -eo pipefail

# Instance-major variant of run_infer.sh
# Instead of processing instances in batches (where stragglers block the whole batch),
# this schedules work instance-by-instance:
#   instance1_run1..runN, instance2_run1..runN, ...
# Docker cleanup is handled per-instance inside the Python script, so no background
# cleanup process is needed.

# Usage:
#   ./run_infer_instance_major.sh <MODEL_CONFIG> <COMMIT_HASH> <AGENT> <EVAL_LIMIT> \
#       <MAX_ITER> <NUM_WORKERS> <DATASET> <SPLIT> <N_RUNS> <MODE> \
#       [EVAL_OUTPUT_DIR] [PREPARE_ENV_MAX_WORKERS]
#
# Arguments:
#   MODEL_CONFIG             LLM config name (e.g. llm.eval)
#   COMMIT_HASH              Git commit hash for version tracking
#   AGENT                    Agent class (default: CodeActAgent)
#   EVAL_LIMIT               Max number of instances to evaluate (optional)
#   MAX_ITER                 Max iterations per instance (default: 100)
#   NUM_WORKERS              Number of parallel workers (default: 1)
#   DATASET                  Dataset name (default: princeton-nlp/SWE-bench_Lite)
#   SPLIT                    Dataset split (default: test)
#   N_RUNS                   Number of runs per instance (default: 1)
#   MODE                     Evaluation mode: swe, swt, or swt-ci (default: swe)
#   EVAL_OUTPUT_DIR          Output directory (default: evaluation/evaluation_outputs/outputs)
#   PREPARE_ENV_MAX_WORKERS  Max concurrent env preparations (default: 2)
#
# Environment variables:
#   EXP_NAME                 Experiment name appended to eval note
#   USE_HINT_TEXT            Use hint text (default: false)
#   RUN_WITH_BROWSING        Enable browsing (default: false)
#   EVAL_CONDENSER           Condenser config name
#   SKIP_RUNS                Comma-separated run IDs to skip (e.g. "1,3")
#
# Example:
#   bash ./evaluation/benchmarks/swe_bench_optimized/scripts/run_infer_instance_major.sh llm.eval_policy_traj_128k_swegym_634i_qwen2-5_coder_14b_full_128k_megatron HEAD CodeActAgent 500 100 32 princeton-nlp/SWE-bench_Verified test

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

if [ -n "$PREPARE_ENV_MAX_WORKERS" ] && [ "$PREPARE_ENV_MAX_WORKERS" -gt 0 ] 2>/dev/null; then
    export OH_RUNTIME_PREPARE_MAX_CONCURRENCY=$PREPARE_ENV_MAX_WORKERS
fi


if [ -z "$NUM_WORKERS" ]; then
  NUM_WORKERS=1
  echo "Number of workers not specified, use default $NUM_WORKERS"
fi
checkout_eval_branch

if [ -z "$AGENT" ]; then
  echo "Agent not specified, use default CodeActAgent"
  AGENT="CodeActAgent"
fi

if [ -z "$MAX_ITER" ]; then
  echo "MAX_ITER not specified, use default 100"
  MAX_ITER=100
fi

if [ -z "$RUN_WITH_BROWSING" ]; then
  echo "RUN_WITH_BROWSING not specified, use default false"
  RUN_WITH_BROWSING=false
fi


if [ -z "$DATASET" ]; then
  echo "DATASET not specified, use default princeton-nlp/SWE-bench_Lite"
  DATASET="princeton-nlp/SWE-bench_Lite"
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
export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=true
export OH_RUNTIME_RUNTIME_IMAGE_REPO="docker.io/mmr1115/openhands-runtime"

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

  COMMAND="poetry run python evaluation/benchmarks/swe_bench_optimized/run_infer_instance_major.py \
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
echo "### (each instance completes all runs before moving to the next) ###"
run_eval "$EVAL_NOTE" "$N_RUNS"

checkout_original_branch
