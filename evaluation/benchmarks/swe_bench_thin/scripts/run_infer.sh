#!/usr/bin/env bash
set -eo pipefail

source "evaluation/utils/version_control.sh"

MODEL_CONFIG=${1:-"llm.eval_glm5_fp8_t0"}
COMMIT_HASH=${2:-"HEAD"}
AGENT=${3:-"CodeActAgent"}
EVAL_LIMIT=${4:-"500"}
MAX_ITER=${5:-"100"}
NUM_WORKERS=${6:-"4"}
DATASET=${7:-"princeton-nlp/SWE-bench_Verified"}
SPLIT=${8:-"test"}
MODE=${9:-"swe"}
EVAL_OUTPUT_DIR=${10:-${EVAL_OUTPUT_DIR:-"evaluation/evaluation_outputs/outputs_thin"}}

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

if [ -z "$DATASET" ]; then
  echo "DATASET not specified, use default princeton-nlp/SWE-bench_Verified"
  DATASET="princeton-nlp/SWE-bench_Verified"
fi

if [ -z "$SPLIT" ]; then
  echo "SPLIT not specified, use default test"
  SPLIT="test"
fi

if [ -z "$MODE" ]; then
  MODE="swe"
  echo "MODE not specified, use default $MODE"
fi

# Force thin_docker runtime
export RUNTIME=thin_docker
# No browsing in thin runtime
export RUN_WITH_BROWSING=false
export DEFAULT_RUNTIME_RESOURCE_FACTOR=2

get_openhands_version

echo "======================================"
echo "SWE-bench Thin Docker Runtime"
echo "======================================"
echo "AGENT: $AGENT"
echo "OPENHANDS_VERSION: $OPENHANDS_VERSION"
echo "MODEL_CONFIG: $MODEL_CONFIG"
echo "DATASET: $DATASET"
echo "SPLIT: $SPLIT"
echo "MAX_ITER: $MAX_ITER"
echo "NUM_WORKERS: $NUM_WORKERS"
echo "COMMIT_HASH: $COMMIT_HASH"
echo "MODE: $MODE"
echo "RUNTIME: thin_docker"
echo "EVAL_OUTPUT_DIR: $EVAL_OUTPUT_DIR"
echo "======================================"

# Default to NOT use Hint
if [ -z "$USE_HINT_TEXT" ]; then
  export USE_HINT_TEXT=false
fi
echo "USE_HINT_TEXT: $USE_HINT_TEXT"

EVAL_NOTE="$OPENHANDS_VERSION-thin"
if [ "$USE_HINT_TEXT" = false ]; then
  EVAL_NOTE="$EVAL_NOTE-no-hint"
fi
if [ -n "$EXP_NAME" ]; then
  EVAL_NOTE="$EVAL_NOTE-$EXP_NAME"
fi
if [ "$MODE" != "swe" ]; then
  EVAL_NOTE="${EVAL_NOTE}-${MODE}"
fi

function run_eval() {
  local eval_note="${1}"

  COMMAND="poetry run python evaluation/benchmarks/swe_bench_thin/run_infer.py \
    --agent-cls $AGENT \
    --llm-config $MODEL_CONFIG \
    --max-iterations $MAX_ITER \
    --eval-num-workers $NUM_WORKERS \
    --eval-note $eval_note \
    --eval-output-dir $EVAL_OUTPUT_DIR \
    --dataset $DATASET \
    --split $SPLIT \
    --mode $MODE"

  if [ -n "$EVAL_LIMIT" ]; then
    echo "EVAL_LIMIT: $EVAL_LIMIT"
    COMMAND="$COMMAND --eval-n-limit $EVAL_LIMIT"
  fi

  eval $COMMAND
}

unset SANDBOX_ENV_GITHUB_TOKEN
run_eval "$EVAL_NOTE"

checkout_original_branch
