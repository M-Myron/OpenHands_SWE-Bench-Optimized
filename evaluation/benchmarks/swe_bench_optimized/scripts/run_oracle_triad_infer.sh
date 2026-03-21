#!/usr/bin/env bash
# run_oracle_triad_infer.sh - launcher for Oracle-Triad SWE-bench evaluation.
#
# Usage:
#   bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
#     <MODEL_CONFIG> <COMMIT_HASH> [AGENT] [EVAL_LIMIT] [MAX_ITER] \
#     [NUM_WORKERS] [DATASET] [SPLIT] [N_RUNS]

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
  DATASET="princeton-nlp/SWE-bench_Lite"
fi

if [ -z "$SPLIT" ]; then
  SPLIT="test"
fi

if [ -z "$N_RUNS" ]; then
  N_RUNS=1
fi

if [ -z "$BLINDED_DEBUGGER_NUM_CANDIDATES" ]; then
  BLINDED_DEBUGGER_NUM_CANDIDATES=3
fi

if [ -z "$ORACLE_PLANNER_MAX_RETRIES" ]; then
  ORACLE_PLANNER_MAX_RETRIES=2
fi

if [ -z "$ORACLE_PLANNER_LLM_CONFIG" ]; then
  ORACLE_PLANNER_LLM_CONFIG="oracle_planner"
fi

if [ -z "$ORACLE_PROPOSAL_CRITIC_LLM_CONFIG" ]; then
  ORACLE_PROPOSAL_CRITIC_LLM_CONFIG="blinded_critic"
fi

if [ -z "$ORACLE_PLANNER_SAVE_PROMPTS" ]; then
  ORACLE_PLANNER_SAVE_PROMPTS=0
fi

if [ -z "$ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS" ]; then
  ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS=0
fi

if [ -z "$RUN_WITH_BROWSING" ]; then
  RUN_WITH_BROWSING=false
fi

export BLINDED_DEBUGGER_NUM_CANDIDATES
export ORACLE_PLANNER_MAX_RETRIES
export ORACLE_PLANNER_LLM_CONFIG
export ORACLE_PROPOSAL_CRITIC_LLM_CONFIG
export ORACLE_PLANNER_SAVE_PROMPTS
export ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS
export RUN_WITH_BROWSING
export DEFAULT_RUNTIME_RESOURCE_FACTOR=2
export INSTANCE_IDS="django__django-12663"

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
echo "  CANDIDATES/STEP:          $BLINDED_DEBUGGER_NUM_CANDIDATES"
echo "  PLANNER_RETRIES:          $ORACLE_PLANNER_MAX_RETRIES"
echo "  PLANNER_LLM_CONFIG:       $ORACLE_PLANNER_LLM_CONFIG"
echo "  PROPOSAL_CRITIC_CONFIG:   $ORACLE_PROPOSAL_CRITIC_LLM_CONFIG"
echo "  EVAL_NOTE:                $EVAL_NOTE"
echo "  OPENHANDS_VERSION:        $OPENHANDS_VERSION"
echo "  COMMIT_HASH:              $COMMIT_HASH"
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
