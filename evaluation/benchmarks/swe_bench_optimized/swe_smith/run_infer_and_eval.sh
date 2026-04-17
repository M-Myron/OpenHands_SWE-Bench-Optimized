#!/usr/bin/env bash
set -eo pipefail

# Combined SWE-smith inference + evaluation script.
# Runs inference then automatically evaluates.
#
# Usage:
#   ./run_infer_and_eval.sh <MODEL_CONFIG> <COMMIT_HASH> <AGENT> <EVAL_LIMIT> \
#       <MAX_ITER> <NUM_WORKERS> [N_RUNS]
#
# Example:
#   bash evaluation/benchmarks/swe_bench_optimized/swe_smith/run_infer_and_eval.sh \
#       llm.eval HEAD CodeActAgent 100 100 8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================================="
echo "Step 1: Running SWE-smith inference"
echo "=============================================================="

INFER_LOG=$(mktemp /tmp/swesmith_infer_log.XXXXXX)
trap "rm -f $INFER_LOG" EXIT

bash "$SCRIPT_DIR/run_infer.sh" "$@" 2>&1 | tee "$INFER_LOG"
INFER_EXIT=${PIPESTATUS[0]}

if [ "$INFER_EXIT" -ne 0 ]; then
    echo "ERROR: Inference failed with exit code $INFER_EXIT"
    exit $INFER_EXIT
fi

# Extract output file path(s) from log
OUTPUT_FILES=$(grep -oP '### OUTPUT FILE(?:\s+FOR RUN \d+)?: \K[^\s]+' "$INFER_LOG" | sort -u)

if [ -z "$OUTPUT_FILES" ]; then
    echo "WARNING: Could not find OUTPUT FILE path in log. Looking for output.jsonl..."
    EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"evaluation/evaluation_outputs/outputs"}
    OUTPUT_FILES=$(find "$EVAL_OUTPUT_DIR" -path "*SWE-smith*" -name "output.jsonl" -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -z "$OUTPUT_FILES" ]; then
        echo "ERROR: Could not locate any output.jsonl for SWE-smith."
        exit 1
    fi
fi

echo ""
echo "=============================================================="
echo "Step 2: Running SWE-smith evaluation"
echo "=============================================================="

for OUTPUT_FILE in $OUTPUT_FILES; do
    if [ ! -f "$OUTPUT_FILE" ]; then
        echo "WARNING: $OUTPUT_FILE does not exist, skipping."
        continue
    fi
    echo ""
    echo "Evaluating: $OUTPUT_FILE"
    echo "--------------------------------------------------------------"
    bash "$SCRIPT_DIR/eval_infer.sh" "$OUTPUT_FILE" "SWE-bench/SWE-smith" "train"
done

echo ""
echo "=============================================================="
echo "SWE-smith inference + evaluation complete!"
echo "=============================================================="
