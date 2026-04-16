#!/usr/bin/env bash
set -eo pipefail

# Oracle-Guided V2 instance-major inference using ThinDockerRuntime.
# Wraps swe_bench_optimized/run_infer_oracle_guided_v2_instance_major.py with RUNTIME=thin_docker.
#
# Usage:
#   bash evaluation/benchmarks/swe_bench_thin/scripts/run_oracle_guided_v2_infer_instance_major.sh \
#     <MODEL_CONFIG> <COMMIT_HASH> [AGENT] [EVAL_LIMIT] [MAX_ITER] \
#     [NUM_WORKERS] [DATASET] [SPLIT] [N_RUNS]

source "evaluation/utils/version_control.sh"

MODEL_CONFIG=${1:-"llm.eval_glm5_fp8_t0"}
COMMIT_HASH=${2:-"HEAD"}
AGENT=${3:-OracleGuidedV2CodeActAgent}
EVAL_LIMIT=$4
MAX_ITER=${5:-100}
NUM_WORKERS=${6:-4}
DATASET=${7:-SWE-Gym/SWE-Gym}
SPLIT=${8:-train}
N_RUNS=${9:-1}

checkout_eval_branch

# Force thin_docker runtime
export RUNTIME=thin_docker
export RUN_WITH_BROWSING=false
export DEFAULT_RUNTIME_RESOURCE_FACTOR=2
export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=true

# ---------------------------------------------------------------------------
# V2 guided agent env vars (same as optimized version)
# ---------------------------------------------------------------------------
_GUIDED_V2_VARS=(
  GUIDED_V2_NUM_CANDIDATES
  GUIDED_V2_MAX_RETRIES
  GUIDED_V2_PLANNER_HISTORY_NEAR_WINDOW
  GUIDED_V2_PLANNER_LLM_CONFIG
  GUIDED_V2_LEAKAGE_CRITIC_LLM_CONFIG
  GUIDED_V2_SUFFICIENCY_CRITIC_LLM_CONFIG
  GUIDED_V2_SAVE_PLANNER_PROMPTS
  GUIDED_V2_SAVE_CRITIC_PROMPTS
  GUIDED_V2_PLANNER_JSON_PARSE_MAX_RETRIES
  GUIDED_V2_LEAKAGE_CRITIC_JSON_PARSE_MAX_RETRIES
  GUIDED_V2_SUFFICIENCY_CRITIC_JSON_PARSE_MAX_RETRIES
)

if [ "${GUIDED_V2_CLEAN_ENV:-0}" = "1" ]; then
  for _v in "${_GUIDED_V2_VARS[@]}"; do unset "$_v"; done
fi

export ORACLE_GUIDED_V2_CONFIG="${ORACLE_GUIDED_V2_CONFIG:-/home/v-murongma/code/OpenHands_SWE-Bench-Optimized/openhands/agenthub/oracle_guided_v2_codeact_agent/guided_config.yaml}"
export ORACLE_PREPROCESS_DIR="${ORACLE_PREPROCESS_DIR:-/home/v-murongma/code/OpenHands_SWE-Bench-Optimized/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/preprocess/swegym_v6_phase1}"
export ORACLE_GRAPH_FILTER_JSON="${ORACLE_GRAPH_FILTER_JSON:-}"

_export_if_set() { [ -n "${!1+x}" ] && export "$1" || true; }
for _v in "${_GUIDED_V2_VARS[@]}"; do _export_if_set "$_v"; done
_export_if_set ORACLE_GUIDED_V2_CONFIG

get_openhands_version

USE_HINT_TEXT=${USE_HINT_TEXT:-false}
export USE_HINT_TEXT

EVAL_NOTE="${OPENHANDS_VERSION}-thin-oracle-guided-v2"
[ -n "$EXP_NAME" ] && EVAL_NOTE="${EVAL_NOTE}-${EXP_NAME}"

echo "========================================================"
echo "  ORACLE GUIDED V2 (THIN DOCKER, INSTANCE-MAJOR)"
echo "========================================================"
echo "  AGENT:                    $AGENT"
echo "  MODEL_CONFIG:             $MODEL_CONFIG"
echo "  DATASET:                  $DATASET"
echo "  SPLIT:                    $SPLIT"
echo "  MAX_ITER:                 $MAX_ITER"
echo "  NUM_WORKERS:              $NUM_WORKERS"
echo "  N_RUNS:                   $N_RUNS"
echo "  RUNTIME:                  thin_docker"
echo "  ORACLE_PREPROCESS_DIR:    ${ORACLE_PREPROCESS_DIR:-(not set)}"
echo "  ORACLE_GUIDED_V2_CONFIG:  ${ORACLE_GUIDED_V2_CONFIG:-(not set)}"
echo "  EVAL_NOTE:                $EVAL_NOTE"
echo "========================================================"

unset SANDBOX_ENV_GITHUB_TOKEN

function run_inference() {
  local command="poetry run python evaluation/benchmarks/swe_bench_optimized/run_infer_oracle_guided_v2_instance_major.py \
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
