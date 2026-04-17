#!/usr/bin/env bash
set -eo pipefail

# SWE-smith inference script for OpenHands.
# Runs the agent on SWE-smith real PR-mirror instances.
# Output format matches SWE-Gym.
#
# Usage:
#   ./run_infer.sh <MODEL_CONFIG> <COMMIT_HASH> <AGENT> <EVAL_LIMIT> \
#       <MAX_ITER> <NUM_WORKERS> [N_RUNS]
#
# Example:
#   bash evaluation/benchmarks/swe_bench_optimized/swe_smith/run_infer.sh \
#       llm.eval HEAD CodeActAgent 100 100 8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

source "$REPO_ROOT/evaluation/utils/version_control.sh"

MODEL_CONFIG=$1
COMMIT_HASH=$2
AGENT=${3:-"CodeActAgent"}
EVAL_LIMIT=$4
MAX_ITER=${5:-100}
NUM_WORKERS=${6:-1}
N_RUNS=${7:-1}

# Fixed dataset params for SWE-smith
DATASET="SWE-bench/SWE-smith"
SPLIT="train"
MODE="swe"

EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"evaluation/evaluation_outputs/outputs"}

checkout_eval_branch
get_openhands_version

export RUN_WITH_BROWSING=${RUN_WITH_BROWSING:-false}
export USE_HINT_TEXT=${USE_HINT_TEXT:-false}
export DEFAULT_RUNTIME_RESOURCE_FACTOR=${DEFAULT_RUNTIME_RESOURCE_FACTOR:-2}
export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=true

echo "=============================================================="
echo "SWE-smith Inference"
echo "=============================================================="
echo "MODEL_CONFIG: $MODEL_CONFIG"
echo "AGENT: $AGENT"
echo "DATASET: $DATASET"
echo "MAX_ITER: $MAX_ITER"
echo "NUM_WORKERS: $NUM_WORKERS"
echo "N_RUNS: $N_RUNS"
echo "OPENHANDS_VERSION: $OPENHANDS_VERSION"
echo "=============================================================="

EVAL_NOTE="$OPENHANDS_VERSION"
if [ "$USE_HINT_TEXT" = false ]; then
    EVAL_NOTE="$EVAL_NOTE-no-hint"
fi
if [ -n "$EXP_NAME" ]; then
    EVAL_NOTE="$EVAL_NOTE-$EXP_NAME"
fi

COMMAND="poetry run python evaluation/benchmarks/swe_bench_optimized/swe_smith/run_infer.py \
    --agent-cls $AGENT \
    --llm-config $MODEL_CONFIG \
    --max-iterations $MAX_ITER \
    --eval-num-workers $NUM_WORKERS \
    --eval-note $EVAL_NOTE \
    --eval-output-dir $EVAL_OUTPUT_DIR \
    --dataset $DATASET \
    --split $SPLIT \
    --mode $MODE \
    --n-runs $N_RUNS \
    --filter-real-only"

if [ -n "$EVAL_LIMIT" ]; then
    COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
fi

unset SANDBOX_ENV_GITHUB_TOKEN
eval $COMMAND
