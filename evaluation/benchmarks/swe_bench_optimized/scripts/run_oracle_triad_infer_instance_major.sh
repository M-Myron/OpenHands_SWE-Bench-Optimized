#!/usr/bin/env bash
# run_oracle_triad_infer_instance_major.sh
#
# Instance-major Oracle-Triad evaluation.
# Schedules: instance1_run1..runN, instance2_run1..runN, ...
# No batch-wait overhead — workers pick up next instance immediately.
#
# Usage:
#   bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer_instance_major.sh \
#     <MODEL_CONFIG> <COMMIT_HASH> [AGENT] [EVAL_LIMIT] [MAX_ITER] \
#     [NUM_WORKERS] [DATASET] [SPLIT] [N_RUNS]

set -eo pipefail

source "evaluation/utils/version_control.sh"

MODEL_CONFIG=$1
COMMIT_HASH=$2
AGENT=${3:-OracleTriadCodeActAgent}
EVAL_LIMIT=$4
MAX_ITER=${5:-100}
NUM_WORKERS=${6:-1}
DATASET=${7:-SWE-Gym/SWE-Gym}
SPLIT=${8:-train}
N_RUNS=${9:-1}

checkout_eval_branch

# ---------------------------------------------------------------------------
# Triad agent settings (same as run_oracle_triad_infer.sh)
# ---------------------------------------------------------------------------
_TRIAD_VARS=(
  BLINDED_DEBUGGER_NUM_CANDIDATES
  ORACLE_PLANNER_MAX_RETRIES
  ORACLE_PLANNER_HISTORY_WINDOW
  ORACLE_PLANNER_LLM_CONFIG
  ORACLE_PROPOSAL_CRITIC_LLM_CONFIG
  ORACLE_PLANNER_SAVE_PROMPTS
  ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS
  PROPOSAL_VALIDATOR
  VERIFIER_LLM_CONFIG
  VERIFIER_PROGRAMMATIC_ONLY
  VERIFIER_EXTRACTOR_JSON_RETRIES
  ORACLE_PLANNER_JSON_PARSE_MAX_RETRIES
  ORACLE_PROPOSAL_CRITIC_JSON_PARSE_MAX_RETRIES
)

if [ "${TRIAD_CLEAN_ENV:-0}" = "1" ]; then
  for _v in "${_TRIAD_VARS[@]}"; do unset "$_v"; done
  echo "[clean] Cleared all triad env vars."
fi

if [ -z "$ORACLE_TRIAD_CONFIG" ]; then
  _DEFAULT_CONFIG="openhands/agenthub/oracle_triad_codeact_agent/triad_config.default.yaml"
  [ -f "$_DEFAULT_CONFIG" ] && ORACLE_TRIAD_CONFIG="$_DEFAULT_CONFIG"
fi

if [ -z "$ORACLE_PREPROCESS_DIR" ]; then
  _DATASET_SLUG=$(echo "$DATASET" | sed 's|/|__|g')-${SPLIT:-test}
  _AUTO="evaluation/evaluation_outputs/outputs/${_DATASET_SLUG}/preprocess"
  if [ -d "${_AUTO}/swegym_v5" ]; then ORACLE_PREPROCESS_DIR="${_AUTO}/swegym_v5"
  elif [ -d "${_AUTO}/swegym_v3" ]; then ORACLE_PREPROCESS_DIR="${_AUTO}/swegym_v3"
  elif [ -d "$_AUTO" ]; then ORACLE_PREPROCESS_DIR="$_AUTO"
  else ORACLE_PREPROCESS_DIR=""; fi
fi

_export_if_set() { [ -n "${!1+x}" ] && export "$1" || true; }
for _v in "${_TRIAD_VARS[@]}"; do _export_if_set "$_v"; done
_export_if_set ORACLE_TRIAD_CONFIG

export ORACLE_PREPROCESS_DIR
export RUN_WITH_BROWSING=${RUN_WITH_BROWSING:-false}
export DEFAULT_RUNTIME_RESOURCE_FACTOR=2
export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=true
export OH_RUNTIME_RUNTIME_IMAGE_REPO="docker.io/mmr1115/openhands-runtime"
export OH_RUNTIME_PREPARE_MAX_CONCURRENCY=8

# Limit concurrent runtime environment preparation (0/empty = no limit)
if [ -n "$OH_RUNTIME_PREPARE_MAX_CONCURRENCY" ] && [ "$OH_RUNTIME_PREPARE_MAX_CONCURRENCY" -gt 0 ] 2>/dev/null; then
  export OH_RUNTIME_PREPARE_MAX_CONCURRENCY
fi
# Guard against stuck relaunches caused by stale env-prepare lock files.
export OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS=${OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS:-30}
export OH_RUNTIME_PREPARE_TIMEOUT_SECONDS=${OH_RUNTIME_PREPARE_TIMEOUT_SECONDS:-1800}
export OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS=${OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS:-7200}


get_openhands_version

USE_HINT_TEXT=${USE_HINT_TEXT:-false}
export USE_HINT_TEXT

EVAL_NOTE="${OPENHANDS_VERSION}-oracle-triad"
[ -n "$EXP_NAME" ] && EVAL_NOTE="${EVAL_NOTE}-${EXP_NAME}"
[ -n "$EVAL_CONDENSER" ] && EVAL_NOTE="${EVAL_NOTE}-${EVAL_CONDENSER}"

echo "========================================================"
echo "  ORACLE TRIAD EVALUATION (INSTANCE-MAJOR)"
echo "========================================================"
echo "  AGENT:                $AGENT"
echo "  MODEL_CONFIG:         $MODEL_CONFIG"
echo "  DATASET:              $DATASET"
echo "  SPLIT:                $SPLIT"
echo "  MAX_ITER:             $MAX_ITER"
echo "  NUM_WORKERS:          $NUM_WORKERS"
echo "  N_RUNS:               $N_RUNS"
echo "  ORACLE_PREPROCESS_DIR: ${ORACLE_PREPROCESS_DIR:-(not set)}"
echo "  ORACLE_TRIAD_CONFIG:  ${ORACLE_TRIAD_CONFIG:-(not set)}"
echo "  EVAL_NOTE:            $EVAL_NOTE"
echo "========================================================"

unset SANDBOX_ENV_GITHUB_TOKEN

function run_inference() {
  local command="poetry run python evaluation/benchmarks/swe_bench_optimized/run_infer_oracle_triad_instance_major.py \
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

function docker_cleanup() {
  echo "### Cleaning up docker artifacts... ###"
  docker ps -q --filter "name=openhands-runtime-" | xargs -r docker stop 2>/dev/null || true
  docker ps -aq --filter "name=openhands-runtime-" | xargs -r docker rm 2>/dev/null || true
  docker images --format "{{.Repository}}:{{.Tag}}" \
    | grep -E "^(docker\.io/xingyaoww/|mmr1115/openhands-runtime|ghcr\.io/openhands/runtime)" \
    | xargs -r docker rmi -f 2>/dev/null || true
  docker image prune -f 2>/dev/null || true
  docker builder prune -f --filter "until=1h" 2>/dev/null || true
  echo "### Docker cleanup done. ###"
}

# Retry loop: re-run inference if it crashes (e.g. transient Docker errors).
# Each retry cleans up stale containers/images first.
# The Python runner tracks completed instances, so retries only process remaining work.
while true; do
  echo "### Running Oracle-Triad inference (instance-major)... ###"
  run_inference
  INFER_STATUS=$?

  docker_cleanup

  if [ $INFER_STATUS -eq 0 ]; then
    echo "### Inference completed successfully. ###"
    break
  else
    echo "### Inference failed (exit=$INFER_STATUS). Retrying after cleanup... ###"
  fi
done

checkout_original_branch
