#!/usr/bin/env bash
# run_oracle_triad_infer.sh - launcher for Oracle-Triad SWE-bench evaluation.
#
# Usage:
#   bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
#     <MODEL_CONFIG> <COMMIT_HASH> [AGENT] [EVAL_LIMIT] [MAX_ITER] \
#     [NUM_WORKERS] [DATASET] [SPLIT] [N_RUNS]
# Example:
# '''
# bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
#   llm.eval_glm5_fp8_t0 HEAD OracleTriadCodeActAgent 1 100 1
# '''

set -eo pipefail

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

if [ -z "$NUM_WORKERS" ]; then
  NUM_WORKERS=1
fi

checkout_eval_branch

if [ -z "$AGENT" ]; then
  AGENT="OracleTriadCodeActAgent"
fi

if [ -z "$MAX_ITER" ]; then
  MAX_ITER=100
fi

if [ -z "$DATASET" ]; then
  # DATASET="princeton-nlp/SWE-bench_Lite"
  DATASET="SWE-Gym/SWE-Gym"
fi

if [ -z "$SPLIT" ]; then
  # SPLIT="test"
  SPLIT="train"
fi

if [ -z "$N_RUNS" ]; then
  N_RUNS=1
fi

# ---------------------------------------------------------------------------
# Triad agent settings
# ---------------------------------------------------------------------------
# Two ways to configure:
#   1. YAML config:  ORACLE_TRIAD_CONFIG=path/to/config.yaml  (recommended)
#   2. Env vars:     BLINDED_DEBUGGER_NUM_CANDIDATES=3         (one-off overrides)
#
# Priority: explicit env var  >  YAML config  >  Python defaults
#
# PROBLEM: env vars from a previous shell session persist silently.
# If you're unsure what's active, use TRIAD_CLEAN_ENV=1 to clear everything:
#   TRIAD_CLEAN_ENV=1 bash run_oracle_triad_infer.sh ...
# Or use TRIAD_CLEAN_ENV=1 together with a YAML config for full control:
#   TRIAD_CLEAN_ENV=1 ORACLE_TRIAD_CONFIG=my.yaml bash run_oracle_triad_infer.sh ...
# ---------------------------------------------------------------------------

# List of triad-controlled env vars (Python TriadConfig handles their defaults)
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

# Optional: clear all triad env vars for a guaranteed clean slate.
if [ "${TRIAD_CLEAN_ENV:-0}" = "1" ]; then
  for _v in "${_TRIAD_VARS[@]}"; do unset "$_v"; done
  echo "[clean] Cleared all triad env vars. Using YAML config / Python defaults only."
fi

# Default ORACLE_TRIAD_CONFIG to the bundled config file if not set
if [ -z "$ORACLE_TRIAD_CONFIG" ]; then
  _DEFAULT_CONFIG="openhands/agenthub/oracle_triad_codeact_agent/triad_config.default.yaml"
  if [ -f "$_DEFAULT_CONFIG" ]; then
    ORACLE_TRIAD_CONFIG="$_DEFAULT_CONFIG"
  fi
fi

# Auto-detect preprocess directory if not set
if [ -z "$ORACLE_PREPROCESS_DIR" ]; then
  _DATASET_SLUG=$(echo "$DATASET" | sed 's|/|__|g')-${SPLIT:-test}
  _AUTO_PREPROCESS="evaluation/evaluation_outputs/outputs/${_DATASET_SLUG}/preprocess"
  if [ -d "${_AUTO_PREPROCESS}/swegym_v5" ]; then
    ORACLE_PREPROCESS_DIR="${_AUTO_PREPROCESS}/swegym_v5"
  elif [ -d "${_AUTO_PREPROCESS}/swegym_v3" ]; then
    ORACLE_PREPROCESS_DIR="${_AUTO_PREPROCESS}/swegym_v3"
  elif [ -d "$_AUTO_PREPROCESS" ]; then
    ORACLE_PREPROCESS_DIR="$_AUTO_PREPROCESS"
  else
    ORACLE_PREPROCESS_DIR=""
  fi
fi

# Export triad env vars — only if the user explicitly set them.
# Unset vars are left for Python TriadConfig to fill from YAML or defaults.
_export_if_set() { [ -n "${!1+x}" ] && export "$1" || true; }
for _v in "${_TRIAD_VARS[@]}"; do _export_if_set "$_v"; done
_export_if_set ORACLE_TRIAD_CONFIG

# These are always exported (not triad-config-controlled)
export OH_RUNTIME_RUNTIME_IMAGE_REPO="docker.io/mmr1115/openhands-runtime"
export ORACLE_PREPROCESS_DIR
export RUN_WITH_BROWSING=${RUN_WITH_BROWSING:-false}
export DEFAULT_RUNTIME_RESOURCE_FACTOR=2
# export INSTANCE_IDS="django__django-12663"
# export INSTANCE_IDS="bokeh__bokeh-12779"
# export INSTANCE_IDS="Project-MONAI__MONAI-1012"
export INSTANCE_IDS="getmoto__moto-6857"

get_openhands_version

USE_HINT_TEXT=${USE_HINT_TEXT:-false}
export USE_HINT_TEXT

EVAL_NOTE="${OPENHANDS_VERSION}-oracle-triad"
if [ -n "$EXP_NAME" ]; then
  EVAL_NOTE="${EVAL_NOTE}-${EXP_NAME}"
fi

if [ -n "$EVAL_CONDENSER" ]; then
  EVAL_NOTE="${EVAL_NOTE}-${EVAL_CONDENSER}"
fi

echo "========================================================"
echo "  ORACLE TRIAD EVALUATION"
echo "========================================================"
echo "  AGENT:                    $AGENT"
echo "  MODEL_CONFIG:             $MODEL_CONFIG"
echo "  DATASET:                  $DATASET"
echo "  SPLIT:                    $SPLIT"
echo "  MAX_ITER:                 $MAX_ITER"
echo "  NUM_WORKERS:              $NUM_WORKERS"
echo "  N_RUNS:                   $N_RUNS"
echo "  ORACLE_PREPROCESS_DIR:    ${ORACLE_PREPROCESS_DIR:-(not set)}"
echo "  ORACLE_TRIAD_CONFIG:      ${ORACLE_TRIAD_CONFIG:-(not set)}"
echo "  EVAL_NOTE:                $EVAL_NOTE"
echo "  OPENHANDS_VERSION:        $OPENHANDS_VERSION"
echo "  COMMIT_HASH:              $COMMIT_HASH"
echo "--------------------------------------------------------"
echo "  Triad env var overrides (unset = YAML/default):"
_any_set=0
for _v in "${_TRIAD_VARS[@]}"; do
  if [ -n "${!_v+x}" ]; then
    echo "    $_v=${!_v}"
    _any_set=1
  fi
done
if [ "$_any_set" = "0" ]; then
  echo "    (none — all from YAML config or Python defaults)"
fi
echo "========================================================"

(
  while true; do
    sleep 1800
    docker container prune -f 2>/dev/null || true
    docker image prune -a -f --filter "until=30m" 2>/dev/null || true
    docker builder prune -f --filter "until=30m" 2>/dev/null || true
  done
) &
CLEANUP_PID=$!
trap "kill $CLEANUP_PID 2>/dev/null || true" EXIT INT TERM

unset SANDBOX_ENV_GITHUB_TOKEN

COMMAND="poetry run python evaluation/benchmarks/swe_bench_optimized/run_infer_oracle_triad.py \
  --agent-cls $AGENT \
  --llm-config $MODEL_CONFIG \
  --max-iterations $MAX_ITER \
  --eval-num-workers $NUM_WORKERS \
  --eval-note $EVAL_NOTE \
  --dataset $DATASET \
  --split $SPLIT \
  --mode swe \
  --n-runs $N_RUNS"

if [ -n "$EVAL_LIMIT" ]; then
  COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
fi

if [ -n "$INSTANCE_IDS" ]; then
  COMMAND="$COMMAND --instance-ids $INSTANCE_IDS"
fi

eval $COMMAND

kill $CLEANUP_PID 2>/dev/null || true
trap - EXIT INT TERM

checkout_original_branch
