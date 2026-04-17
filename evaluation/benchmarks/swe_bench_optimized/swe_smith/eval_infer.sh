#!/usr/bin/env bash
set -eo pipefail

# SWE-smith evaluation script for OpenHands.
# Evaluates agent predictions on SWE-smith instances.
#
# Usage:
#   ./eval_infer.sh <OUTPUT_FILE> [DATASET] [SPLIT]
#
# Example:
#   bash evaluation/benchmarks/swe_bench_optimized/swe_smith/eval_infer.sh \
#       evaluation/evaluation_outputs/outputs/SWE-bench__SWE-smith-train/CodeActAgent/run_name/output.jsonl

PROCESS_FILEPATH=$1
if [ -z "$PROCESS_FILEPATH" ]; then
    echo "Error: Usage: ./eval_infer.sh <output_file> [dataset] [split]"
    exit 1
fi

if [ ! -f "$PROCESS_FILEPATH" ]; then
    echo "Error: $PROCESS_FILEPATH does not exist"
    exit 1
fi

DATASET=${2:-"SWE-bench/SWE-smith"}
SPLIT=${3:-"train"}

echo "=============================================================="
echo "SWE-smith Evaluation"
echo "=============================================================="
echo "INPUT: $PROCESS_FILEPATH"
echo "DATASET: $DATASET"
echo "SPLIT: $SPLIT"
echo "=============================================================="

PROCESS_FILEPATH=$(realpath "$PROCESS_FILEPATH")

poetry run python evaluation/benchmarks/swe_bench_optimized/swe_smith/eval_infer.py \
    --input-file "$PROCESS_FILEPATH" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --eval-num-workers ${EVAL_NUM_WORKERS:-8}
