#!/usr/bin/env bash
set -eo pipefail

# Oracle-Guided V1 instance-major inference using ThinDockerRuntime.
# Wraps swe_bench_optimized/run_infer_oracle_guided_instance_major.py with RUNTIME=thin_docker.
#
# Usage:
#   bash evaluation/benchmarks/swe_bench_thin/scripts/run_oracle_guided_infer_instance_major.sh \
#     <MODEL_CONFIG> <COMMIT_HASH> [AGENT] [EVAL_LIMIT] [MAX_ITER] \
#     [NUM_WORKERS] [DATASET] [SPLIT] [N_RUNS]

source "evaluation/utils/version_control.sh"

MODEL_CONFIG=${1:-"llm.eval_glm5_fp8_t0"}
COMMIT_HASH=${2:-"HEAD"}
AGENT=${3:-OracleGuidedCodeActAgent}
EVAL_LIMIT=$4
MAX_ITER=${5:-100}
NUM_WORKERS=${6:-4}
DATASET=${7:-SWE-Gym/SWE-Gym}
SPLIT=${8:-train}
N_RUNS=${9:-1}
PREPARE_ENV_MAX_WORKERS=${10:-${OH_RUNTIME_PREPARE_MAX_CONCURRENCY:-8}}

if [ -n "$PREPARE_ENV_MAX_WORKERS" ] && [ "$PREPARE_ENV_MAX_WORKERS" -gt 0 ] 2>/dev/null; then
    export OH_RUNTIME_PREPARE_MAX_CONCURRENCY=$PREPARE_ENV_MAX_WORKERS
fi

checkout_eval_branch

# Force thin_docker runtime
export RUNTIME=thin_docker
export RUN_WITH_BROWSING=false
export DEFAULT_RUNTIME_RESOURCE_FACTOR=2
export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=true

# Guard against stuck relaunches caused by stale env-prepare lock files.
export OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS=${OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS:-30}
export OH_RUNTIME_PREPARE_TIMEOUT_SECONDS=${OH_RUNTIME_PREPARE_TIMEOUT_SECONDS:-0}
export OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS=${OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS:-7200}

# ---------------------------------------------------------------------------
# Guided agent env vars (same as optimized version)
# ---------------------------------------------------------------------------
_GUIDED_VARS=(
  GUIDED_NUM_CANDIDATES
  GUIDED_PLANNER_MAX_RETRIES
  GUIDED_PLANNER_HISTORY_NEAR_WINDOW
  GUIDED_PLANNER_LLM_CONFIG
  GUIDED_CRITIC_LLM_CONFIG
  GUIDED_SAVE_PLANNER_PROMPTS
  GUIDED_SAVE_CRITIC_PROMPTS
  GUIDED_PLANNER_JSON_PARSE_MAX_RETRIES
  GUIDED_CRITIC_JSON_PARSE_MAX_RETRIES
)

if [ "${GUIDED_CLEAN_ENV:-0}" = "1" ]; then
  for _v in "${_GUIDED_VARS[@]}"; do unset "$_v"; done
fi

export ORACLE_GUIDED_CONFIG="${ORACLE_GUIDED_CONFIG:-/home/v-murongma/code/OpenHands_SWE-Bench-Optimized/openhands/agenthub/oracle_guided_codeact_agent/guided_config.yaml}"
export ORACLE_PREPROCESS_DIR="${ORACLE_PREPROCESS_DIR:-/home/v-murongma/code/OpenHands_SWE-Bench-Optimized/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/preprocess/swegym_v6_phase1}"
export ORACLE_GRAPH_FILTER_JSON="${ORACLE_GRAPH_FILTER_JSON:-/home/v-murongma/code/OpenHands_SWE-Bench-Optimized/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/preprocess/swegym_v6_filter.json}"

_export_if_set() { [ -n "${!1+x}" ] && export "$1" || true; }
for _v in "${_GUIDED_VARS[@]}"; do _export_if_set "$_v"; done
_export_if_set ORACLE_GUIDED_CONFIG

get_openhands_version

USE_HINT_TEXT=${USE_HINT_TEXT:-false}
export USE_HINT_TEXT

EVAL_NOTE="${OPENHANDS_VERSION}-thin-oracle-guided"
[ -n "$EXP_NAME" ] && EVAL_NOTE="${EVAL_NOTE}-${EXP_NAME}"

echo "========================================================"
echo "  ORACLE GUIDED V1 (THIN DOCKER, INSTANCE-MAJOR)"
echo "========================================================"
echo "  AGENT:                    $AGENT"
echo "  MODEL_CONFIG:             $MODEL_CONFIG"
echo "  DATASET:                  $DATASET"
echo "  SPLIT:                    $SPLIT"
echo "  MAX_ITER:                 $MAX_ITER"
echo "  NUM_WORKERS:              $NUM_WORKERS"
echo "  N_RUNS:                   $N_RUNS"
echo "  RUNTIME:                  thin_docker"
if [ -n "$OH_RUNTIME_PREPARE_MAX_CONCURRENCY" ] && [ "$OH_RUNTIME_PREPARE_MAX_CONCURRENCY" -gt 0 ] 2>/dev/null; then
    echo "  PREPARE_CONCURRENCY:      $OH_RUNTIME_PREPARE_MAX_CONCURRENCY"
else
    echo "  PREPARE_CONCURRENCY:      disabled"
fi
echo "  ORACLE_PREPROCESS_DIR:    ${ORACLE_PREPROCESS_DIR:-(not set)}"
echo "  ORACLE_GUIDED_CONFIG:     ${ORACLE_GUIDED_CONFIG:-(not set)}"
echo "  EVAL_NOTE:                $EVAL_NOTE"
echo "========================================================"

unset SANDBOX_ENV_GITHUB_TOKEN

function run_inference() {
  local command="poetry run python evaluation/benchmarks/swe_bench_optimized/run_infer_oracle_guided_instance_major.py \
    --agent-cls $AGENT \
    --llm-config $MODEL_CONFIG \
    --max-iterations $MAX_ITER \
    --eval-num-workers $NUM_WORKERS \
    --eval-note $EVAL_NOTE \
    --dataset $DATASET \
    --split $SPLIT \
    --mode swe \
    --n-runs $N_RUNS"

  [ -n "$EVAL_LIMIT" ] && command="$command --eval-n-limit $EVAL_LIMIT"
  [ -n "$INSTANCE_IDS" ] && command="$command --instance-ids $INSTANCE_IDS"

  eval $command
}

while true; do
  run_inference
  INFER_STATUS=$?
  if [ $INFER_STATUS -eq 0 ]; then
    echo "### Inference completed successfully. ###"
    break
  else
    echo "### Inference failed (exit=$INFER_STATUS). Retrying... ###"
  fi
done

checkout_original_branch
