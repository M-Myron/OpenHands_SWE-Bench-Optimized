#!/usr/bin/env bash
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
MODE=${10}
# Optional: override output directory via 11th arg or EVAL_OUTPUT_DIR env var
EVAL_OUTPUT_DIR=${11:-${EVAL_OUTPUT_DIR:-"evaluation/evaluation_outputs/outputs"}}


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

  # Start periodic Docker cleanup in background
  (
    while true; do
      sleep 1800  # Every 30 minutes (more frequent)
      echo "Running periodic Docker cleanup..."
      # Remove stopped containers
      docker container prune -f 2>/dev/null || true
      # Remove ALL unused images older than 30 minutes (much more aggressive)
      docker image prune -a -f --filter "until=30m" 2>/dev/null || true
      # Also remove build cache to free up space
      docker builder prune -f --filter "until=30m" 2>/dev/null || true
      # Show current usage
      echo "Current Docker usage:"
      docker system df
      echo "Running containers: $(docker ps -q | wc -l)"
      echo "Total images: $(docker images -q | wc -l)"
    done
  ) &
  CLEANUP_PID=$!
  
  # Set up trap to kill cleanup process on script exit or interruption
  trap "kill $CLEANUP_PID 2>/dev/null || true; echo 'Cleanup process stopped'" EXIT INT TERM

  COMMAND="poetry run python evaluation/benchmarks/swe_bench_optimized/run_infer.py \
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

  # Kill the cleanup background process (trap will also handle this)
  kill $CLEANUP_PID 2>/dev/null || true
  trap - EXIT INT TERM  # Remove trap after cleanup
}

unset SANDBOX_ENV_GITHUB_TOKEN # prevent the agent from using the github token to push
if [ -z "$N_RUNS" ]; then
  N_RUNS=1
  echo "N_RUNS not specified, use default $N_RUNS"
fi

# Note: SKIP_RUNS is now handled inside the Python script for better efficiency
# The script will process each instance N_RUNS times before moving to the next instance
# This allows docker images to be reused across runs for the same instance
echo "Running evaluation with N_RUNS=$N_RUNS (processing each instance $N_RUNS times before moving to next)"
run_eval "$EVAL_NOTE" "$N_RUNS"

checkout_original_branch
