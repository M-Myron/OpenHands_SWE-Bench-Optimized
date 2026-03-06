#!/usr/bin/env bash
# run_guided_infer.sh — launcher for the Guided Trajectory experiment.
#
# This mirrors run_infer.sh but targets run_infer_guided.py and
# GuidedCodeActAgent, which includes an in-loop Blinded Critic that
# validates each agent step for reachability and non-leakage.
#
# Usage:
#   bash evaluation/benchmarks/swe_bench_optimized/scripts/run_guided_infer.sh \
#     <MODEL_CONFIG> <COMMIT_HASH> [AGENT] [EVAL_LIMIT] [MAX_ITER] \
#     [NUM_WORKERS] [DATASET] [SPLIT] [N_RUNS]
#
# Extra env vars (all optional):
#   BLINDED_CRITIC_LLM_CONFIG  Name of the [llm.<name>] section in config.toml
#                               to use for the Blinded Critic. Default: blinded_critic
#   BLINDED_CRITIC_MAX_RETRIES Max times the agent may retry a single step
#                               after a critic rejection. Default: 3
#   INSTRUCTION_TEMPLATE_NAME  Override the Jinja2 template for the task
#                               instruction. Default: swe_guided.j2
#
# config.toml requirement — add a section for the Blinded Critic LLM, e.g.:
#
#   [llm.blinded_critic]
#   model = "gpt-4o-mini"
#   api_key = "..."
#   temperature = 0.0
#   max_output_tokens = 1024

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

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

if [ -z "$NUM_WORKERS" ]; then
  NUM_WORKERS=1
  echo "NUM_WORKERS not specified, using default $NUM_WORKERS"
fi

checkout_eval_branch

if [ -z "$AGENT" ]; then
  AGENT="GuidedCodeActAgent"
  echo "AGENT not specified, using default $AGENT"
fi

if [ -z "$MAX_ITER" ]; then
  MAX_ITER=100
  echo "MAX_ITER not specified, using default $MAX_ITER"
fi

if [ -z "$DATASET" ]; then
  DATASET="princeton-nlp/SWE-bench_Lite"
  echo "DATASET not specified, using default $DATASET"
fi

if [ -z "$SPLIT" ]; then
  SPLIT="test"
  echo "SPLIT not specified, using default $SPLIT"
fi

if [ -z "$N_RUNS" ]; then
  N_RUNS=1
  echo "N_RUNS not specified, using default $N_RUNS"
fi

# Blinded Critic settings
if [ -z "$BLINDED_CRITIC_LLM_CONFIG" ]; then
  BLINDED_CRITIC_LLM_CONFIG="blinded_critic"
  echo "BLINDED_CRITIC_LLM_CONFIG not specified, using default '$BLINDED_CRITIC_LLM_CONFIG'"
fi

if [ -z "$BLINDED_CRITIC_MAX_RETRIES" ]; then
  BLINDED_CRITIC_MAX_RETRIES=3
  echo "BLINDED_CRITIC_MAX_RETRIES not specified, using default $BLINDED_CRITIC_MAX_RETRIES"
fi

# BLINDED_CRITIC_SAVE_PROMPTS: set to 1 to save every critic prompt+response
# to {eval_output_dir}/blinded_critic_prompts/{instance_id}/step_NNNN_attempt_MM.txt
# for offline debugging on other LLMs.  Off by default.
if [ -z "$BLINDED_CRITIC_SAVE_PROMPTS" ]; then
  BLINDED_CRITIC_SAVE_PROMPTS=1
fi

if [ -z "$INSTRUCTION_TEMPLATE_NAME" ]; then
  INSTRUCTION_TEMPLATE_NAME="swe_guided.j2"
  echo "INSTRUCTION_TEMPLATE_NAME not specified, using default $INSTRUCTION_TEMPLATE_NAME"
fi

# No browsing in guided mode unless explicitly enabled
if [ -z "$RUN_WITH_BROWSING" ]; then
  RUN_WITH_BROWSING=false
fi

# ---------------------------------------------------------------------------
# Export env vars consumed by the Python script
# ---------------------------------------------------------------------------

export BLINDED_CRITIC_LLM_CONFIG
export BLINDED_CRITIC_MAX_RETRIES
export BLINDED_CRITIC_SAVE_PROMPTS
export INSTRUCTION_TEMPLATE_NAME
export RUN_WITH_BROWSING
export DEFAULT_RUNTIME_RESOURCE_FACTOR=2

# ---------------------------------------------------------------------------
# Eval note
# ---------------------------------------------------------------------------

get_openhands_version

USE_HINT_TEXT=${USE_HINT_TEXT:-false}
export USE_HINT_TEXT

EVAL_NOTE="${OPENHANDS_VERSION}-no-hint-guided"

if [ -n "$EXP_NAME" ]; then
  EVAL_NOTE="${EVAL_NOTE}-${EXP_NAME}"
fi

if [ -n "$EVAL_CONDENSER" ]; then
  EVAL_NOTE="${EVAL_NOTE}-${EVAL_CONDENSER}"
  echo "Using Condenser Config: $EVAL_CONDENSER"
fi

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

echo "========================================================"
echo "  GUIDED TRAJECTORY EVALUATION"
echo "========================================================"
echo "  AGENT:                    $AGENT"
echo "  MODEL_CONFIG:             $MODEL_CONFIG"
echo "  DATASET:                  $DATASET"
echo "  SPLIT:                    $SPLIT"
echo "  MAX_ITER:                 $MAX_ITER"
echo "  NUM_WORKERS:              $NUM_WORKERS"
echo "  N_RUNS:                   $N_RUNS"
echo "  BLINDED_CRITIC_LLM:       $BLINDED_CRITIC_LLM_CONFIG"
echo "  BLINDED_CRITIC_RETRIES:   $BLINDED_CRITIC_MAX_RETRIES"
echo "  BLINDED_CRITIC_PROMPTS:   $BLINDED_CRITIC_SAVE_PROMPTS  (1=save prompts to blinded_critic_prompts/)"
echo "  INSTRUCTION_TEMPLATE:     $INSTRUCTION_TEMPLATE_NAME"
echo "  EVAL_NOTE:                $EVAL_NOTE"
echo "  OPENHANDS_VERSION:        $OPENHANDS_VERSION"
echo "  COMMIT_HASH:              $COMMIT_HASH"
echo "========================================================"

# ---------------------------------------------------------------------------
# Periodic Docker cleanup (background)
# ---------------------------------------------------------------------------

(
  while true; do
    sleep 1800
    echo "Running periodic Docker cleanup..."
    docker container prune -f 2>/dev/null || true
    docker image prune -a -f --filter "until=30m" 2>/dev/null || true
    docker builder prune -f --filter "until=30m" 2>/dev/null || true
    echo "Current Docker usage:"
    docker system df
  done
) &
CLEANUP_PID=$!
trap "kill $CLEANUP_PID 2>/dev/null || true" EXIT INT TERM

# ---------------------------------------------------------------------------
# Build and run the command
# ---------------------------------------------------------------------------

unset SANDBOX_ENV_GITHUB_TOKEN  # prevent agent from pushing

COMMAND="poetry run python evaluation/benchmarks/swe_bench_optimized/run_infer_guided.py \
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
  echo "EVAL_LIMIT: $EVAL_LIMIT"
  COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
fi

eval $COMMAND

kill $CLEANUP_PID 2>/dev/null || true
trap - EXIT INT TERM

checkout_original_branch
