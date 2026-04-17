#!/usr/bin/env bash
# run_oracle_guided_infer.sh - launcher for Oracle Guided SWE-bench evaluation.
#
# Usage:
#   bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_guided_infer.sh \
#     <MODEL_CONFIG> <COMMIT_HASH> [AGENT] [EVAL_LIMIT] [MAX_ITER] \
#     [NUM_WORKERS] [DATASET] [SPLIT] [N_RUNS]
# ORACLE_PREPROCESS_DIR=evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/preprocess/test_v6 \
# bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_guided_infer.sh llm.eval_glm5_fp8_t0 HEAD OracleGuidedCodeActAgent 1 100 1


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

# Auto-detect preprocess directory
if [ -z "$ORACLE_PREPROCESS_DIR" ]; then
  _DATASET_SLUG=$(echo "$DATASET" | sed 's|/|__|g')-${SPLIT:-train}
  _AUTO_PREPROCESS="evaluation/evaluation_outputs/outputs/${_DATASET_SLUG}/preprocess"
  if [ -d "${_AUTO_PREPROCESS}/test_v6" ]; then
    ORACLE_PREPROCESS_DIR="${_AUTO_PREPROCESS}/test_v6"
  elif [ -d "${_AUTO_PREPROCESS}/swegym_v5" ]; then
    ORACLE_PREPROCESS_DIR="${_AUTO_PREPROCESS}/swegym_v5"
  elif [ -d "${_AUTO_PREPROCESS}/swegym_v3" ]; then
    ORACLE_PREPROCESS_DIR="${_AUTO_PREPROCESS}/swegym_v3"
  elif [ -d "$_AUTO_PREPROCESS" ]; then
    ORACLE_PREPROCESS_DIR="$_AUTO_PREPROCESS"
  else
    ORACLE_PREPROCESS_DIR=""
  fi
fi

ORACLE_PREPROCESS_DIR="/home/v-murongma/code/OpenHands_SWE-Bench-Optimized/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/preprocess/swegym_v6"
# INSTANCE_IDS="getmoto__moto-4895"
export ORACLE_GUIDED_CONFIG="/home/v-murongma/code/OpenHands_SWE-Bench-Optimized/openhands/agenthub/oracle_guided_codeact_agent/guided_config.yaml"

# Export guided env vars only if user explicitly set them
_export_if_set() { [ -n "${!1+x}" ] && export "$1" || true; }
for _v in "${_GUIDED_VARS[@]}"; do _export_if_set "$_v"; done
_export_if_set ORACLE_GUIDED_CONFIG

# Always exported
export OH_RUNTIME_RUNTIME_IMAGE_REPO="docker.io/mmr1115/openhands-runtime"
export ORACLE_PREPROCESS_DIR
export ORACLE_GRAPH_FILTER_JSON="${ORACLE_GRAPH_FILTER_JSON:-}"
export RUN_WITH_BROWSING=${RUN_WITH_BROWSING:-false}
export DEFAULT_RUNTIME_RESOURCE_FACTOR=2

get_openhands_version

USE_HINT_TEXT=${USE_HINT_TEXT:-false}
export USE_HINT_TEXT

EVAL_NOTE="${OPENHANDS_VERSION}-oracle-guided"
if [ -n "$EXP_NAME" ]; then
  EVAL_NOTE="${EVAL_NOTE}-${EXP_NAME}"
fi

echo "========================================================"
echo "  ORACLE GUIDED EVALUATION"
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

# Background docker cleanup
_cleanup_docker() {
  while true; do
    sleep 1800
    docker system prune -f --volumes 2>/dev/null || true
  done
}
_cleanup_docker &
_CLEANUP_PID=$!
trap "kill $_CLEANUP_PID 2>/dev/null || true" EXIT

# Build instance IDs argument
INSTANCE_IDS_ARG=""
if [ -n "$INSTANCE_IDS" ]; then
  INSTANCE_IDS_ARG="--instance-ids $INSTANCE_IDS"
fi

# Run evaluation
poetry run python evaluation/benchmarks/swe_bench_optimized/run_infer_oracle_guided.py \
  --agent-cls "$AGENT" \
  --llm-config "$MODEL_CONFIG" \
  --max-iterations "$MAX_ITER" \
  --eval-num-workers "$NUM_WORKERS" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --mode swe \
  --n-runs "$N_RUNS" \
  --eval-note "$EVAL_NOTE" \
  ${EVAL_LIMIT:+--eval-n-limit "$EVAL_LIMIT"} \
  $INSTANCE_IDS_ARG
