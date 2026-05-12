#!/usr/bin/env bash
# eval_infer.sh for swe_bench_thin
#
# Evaluates inference results using the thin_docker runtime, which injects a
# lightweight Python executor into raw SWE-bench/SWE-Gym base images — no
# OpenHands overlay build required.
#
# Supports both SWE-bench and SWE-Gym datasets:
#   - SWE-Gym  → xingyaoww/ images  (e.g. xingyaoww/sweb.eval.x86_64.bokeh_s_bokeh-12867)
#   - SWE-bench → xingyaoww/ or swebench/ images depending on EVAL_DOCKER_IMAGE_PREFIX
#
# Usage:
#   ./eval_infer.sh <output_file> [instance_id] [dataset_name] [split]

set -eo pipefail

PROCESS_FILEPATH=$1
INSTANCE_ID=$2
DATASET_NAME=${3:-"princeton-nlp/SWE-bench_Verified"}
SPLIT=${4:-"test"}

if [ -z "$PROCESS_FILEPATH" ]; then
    echo "Error: PROCESS_FILEPATH is empty."
    echo "Usage: ./eval_infer.sh <output_file> [instance_id] [dataset_name] [split]"
    exit 1
fi

if [ ! -f "$PROCESS_FILEPATH" ]; then
    echo "Error: $PROCESS_FILEPATH is not a file"
    exit 1
fi

PROCESS_FILEPATH=$(realpath "$PROCESS_FILEPATH")
FILE_DIR=$(dirname "$PROCESS_FILEPATH")
FILE_NAME=$(basename "$PROCESS_FILEPATH")

# ── Thin-docker runtime config ──
export RUNTIME=thin_docker
export EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED=${EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED:-true}

# Increase Docker client timeout to avoid "Read timed out" errors
# during concurrent image pulls under heavy I/O load.
export DOCKER_CLIENT_TIMEOUT=${DOCKER_CLIENT_TIMEOUT:-300}
export COMPOSE_HTTP_TIMEOUT=${COMPOSE_HTTP_TIMEOUT:-300}

N_PROCESS=${EVAL_MAX_WORKERS:-8}

echo "=============================================================="
echo "SWE-bench Thin Docker Evaluation"
echo "=============================================================="
echo "INPUT:    $FILE_NAME"
echo "DIR:      $FILE_DIR"
echo "DATASET:  $DATASET_NAME"
echo "SPLIT:    $SPLIT"
echo "WORKERS:  $N_PROCESS"
echo "RUNTIME:  thin_docker"
echo "=============================================================="

EVAL_CMD="poetry run python evaluation/benchmarks/swe_bench_thin/eval_infer.py \
    --eval-num-workers $N_PROCESS \
    --input-file $PROCESS_FILEPATH \
    --dataset $DATASET_NAME \
    --split $SPLIT"

echo ""
echo "Running: $EVAL_CMD"
echo ""
eval $EVAL_CMD
