#!/usr/bin/env bash
# run_oracle_guided_infer_instance_major.sh
#
# Instance-major Oracle-Guided evaluation.
# Schedules: instance1_run1..runN, instance2_run1..runN, ...
# No batch-wait overhead — workers pick up next instance immediately.
#
# Usage:
#   bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_guided_infer_instance_major.sh \
#     <MODEL_CONFIG> <COMMIT_HASH> [AGENT] [EVAL_LIMIT] [MAX_ITER] \
#     [NUM_WORKERS] [DATASET] [SPLIT] [N_RUNS]
#   bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_guided_infer_instance_major.sh llm.eval_glm5_fp8_t0 HEAD OracleGuidedCodeActAgent 3000 100 32 SWE-Gym/SWE-Gym train 1

set -eo pipefail

source "evaluation/utils/version_control.sh"

MODEL_CONFIG=$1
COMMIT_HASH=$2
AGENT=${3:-OracleGuidedCodeActAgent}
EVAL_LIMIT=$4
MAX_ITER=${5:-100}
NUM_WORKERS=${6:-1}
DATASET=${7:-SWE-Gym/SWE-Gym}
SPLIT=${8:-train}
N_RUNS=${9:-1}

checkout_eval_branch

# ---------------------------------------------------------------------------
# Guided agent env vars (parallel to triad but with GUIDED_ prefix)
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

# Optional: clear all guided env vars
if [ "${GUIDED_CLEAN_ENV:-0}" = "1" ]; then
  for _v in "${_GUIDED_VARS[@]}"; do unset "$_v"; done
  echo "[clean] Cleared all guided env vars."
fi

export ORACLE_GUIDED_CONFIG="${ORACLE_GUIDED_CONFIG:-}"

# Export guided env vars only if user explicitly set them
_export_if_set() { [ -n "${!1+x}" ] && export "$1" || true; }
for _v in "${_GUIDED_VARS[@]}"; do _export_if_set "$_v"; done
_export_if_set ORACLE_GUIDED_CONFIG

# Always exported
export OH_RUNTIME_RUNTIME_IMAGE_REPO="docker.io/mmr1115/openhands-runtime"
export ORACLE_PREPROCESS_DIR="/home/v-murongma/code/OpenHands_SWE-Bench-Optimized/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/preprocess/swegym_v6_phase1"
export ORACLE_GRAPH_FILTER_JSON="${ORACLE_GRAPH_FILTER_JSON:-/home/v-murongma/code/OpenHands_SWE-Bench-Optimized/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/preprocess/swegym_v6_filter.json}"
export ORACLE_GUIDED_CONFIG="/home/v-murongma/code/OpenHands_SWE-Bench-Optimized/openhands/agenthub/oracle_guided_codeact_agent/guided_config.yaml"
# export INSTANCE_IDS="getmoto__moto-4951"
export RUN_WITH_BROWSING=${RUN_WITH_BROWSING:-false}
export DEFAULT_RUNTIME_RESOURCE_FACTOR=2
export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=true
export OH_RUNTIME_PREPARE_MAX_CONCURRENCY=${OH_RUNTIME_PREPARE_MAX_CONCURRENCY:-12}

# Guard against stuck relaunches caused by stale env-prepare lock files.
export OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS=${OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS:-30}
export OH_RUNTIME_PREPARE_TIMEOUT_SECONDS=${OH_RUNTIME_PREPARE_TIMEOUT_SECONDS:-1800}
export OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS=${OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS:-7200}

get_openhands_version

USE_HINT_TEXT=${USE_HINT_TEXT:-false}
export USE_HINT_TEXT

EVAL_NOTE="${OPENHANDS_VERSION}-oracle-guided"
if [ -n "$EXP_NAME" ]; then
  EVAL_NOTE="${EVAL_NOTE}-${EXP_NAME}"
fi
[ -n "$EVAL_CONDENSER" ] && EVAL_NOTE="${EVAL_NOTE}-${EVAL_CONDENSER}"

echo "========================================================"
echo "  ORACLE GUIDED EVALUATION (INSTANCE-MAJOR)"
echo "========================================================"
echo "  AGENT:                    $AGENT"
echo "  MODEL_CONFIG:             $MODEL_CONFIG"
echo "  DATASET:                  $DATASET"
echo "  SPLIT:                    $SPLIT"
echo "  MAX_ITER:                 $MAX_ITER"
echo "  NUM_WORKERS:              $NUM_WORKERS"
echo "  N_RUNS:                   $N_RUNS"
echo "  ORACLE_PREPROCESS_DIR:    ${ORACLE_PREPROCESS_DIR:-(not set)}"
echo "  ORACLE_GUIDED_CONFIG:     ${ORACLE_GUIDED_CONFIG:-(not set)}"
echo "  EVAL_NOTE:                $EVAL_NOTE"
echo "  OPENHANDS_VERSION:        $OPENHANDS_VERSION"
echo "  COMMIT_HASH:              $COMMIT_HASH"
echo "--------------------------------------------------------"
echo "  Guided env var overrides (unset = YAML/default):"
_any_set=0
for _v in "${_GUIDED_VARS[@]}"; do
  if [ -n "${!_v+x}" ]; then
    echo "    $_v=${!_v}"
    _any_set=1
  fi
done
if [ "$_any_set" = "0" ]; then
  echo "    (none — using YAML config / Python defaults)"
fi
echo "========================================================"

unset SANDBOX_ENV_GITHUB_TOKEN

# Build instance IDs argument
INSTANCE_IDS_ARG=""
if [ -n "$INSTANCE_IDS" ]; then
  INSTANCE_IDS_ARG="--instance-ids $INSTANCE_IDS"
fi

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
  echo "### Running Oracle-Guided inference (instance-major)... ###"
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
