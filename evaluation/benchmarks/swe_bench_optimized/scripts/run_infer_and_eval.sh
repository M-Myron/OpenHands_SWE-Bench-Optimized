#!/usr/bin/env bash
set -eo pipefail

# Combined script: run inference (instance-major) then automatically evaluate.
#
# Usage:
#   ./run_infer_and_eval.sh <MODEL_CONFIG> <COMMIT_HASH> <AGENT> <EVAL_LIMIT> \
#       <MAX_ITER> <NUM_WORKERS> <DATASET> <SPLIT> <N_RUNS> <MODE> \
#       [EVAL_OUTPUT_DIR] [PREPARE_ENV_MAX_WORKERS]
#
# All arguments are passed through to run_infer_instance_major.sh.
# After inference completes, eval_infer.sh is called with the output.jsonl.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Detect which inference script to use based on the AGENT arg ($3) ──
AGENT=${3:-"CodeActAgent"}
if [[ "$AGENT" == *"SweGymLegacy"* ]]; then
    INFER_SCRIPT="$SCRIPT_DIR/run_infer_swegym_legacy_instance_major.sh"
    echo "Detected SweGymLegacy agent, using: run_infer_swegym_legacy_instance_major.sh"
else
    INFER_SCRIPT="$SCRIPT_DIR/run_infer_instance_major.sh"
    echo "Using standard inference: run_infer_instance_major.sh"
fi

# ── Step 1: Run inference, tee stdout so we can capture the output path ──
INFER_LOG=$(mktemp /tmp/infer_log.XXXXXX)
trap "rm -f $INFER_LOG" EXIT

echo "=============================================================="
echo "Step 1: Running inference (instance-major)"
echo "=============================================================="
bash "$INFER_SCRIPT" "$@" 2>&1 | tee "$INFER_LOG"
INFER_EXIT=${PIPESTATUS[0]}

if [ "$INFER_EXIT" -ne 0 ]; then
    echo "ERROR: Inference failed with exit code $INFER_EXIT"
    exit $INFER_EXIT
fi

# ── Step 2: Extract output file path(s) from inference log ──
# The Python script prints lines like:
#   ### OUTPUT FILE: <path>/output.jsonl ###
#   ### OUTPUT FILE FOR RUN <N>: <path>/output.jsonl ###
OUTPUT_FILES=$(grep -oP '### OUTPUT FILE(?:\s+FOR RUN \d+)?: \K[^\s]+' "$INFER_LOG" | sort -u)

if [ -z "$OUTPUT_FILES" ]; then
    echo "ERROR: Could not find OUTPUT FILE path in inference log."
    echo "Trying fallback: searching for output.jsonl in eval output dir..."

    EVAL_OUTPUT_DIR=${11:-${EVAL_OUTPUT_DIR:-"evaluation/evaluation_outputs/outputs"}}
    DATASET=${7:-"princeton-nlp/SWE-bench_Lite"}
    # Find the most recently modified output.jsonl
    OUTPUT_FILES=$(find "$EVAL_OUTPUT_DIR" -path "*${DATASET##*/}*" -name "output.jsonl" -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-)

    if [ -z "$OUTPUT_FILES" ]; then
        echo "ERROR: Could not locate any output.jsonl. Aborting evaluation."
        exit 1
    fi
fi

# ── Step 3: Run evaluation for each output file ──
DATASET=${7:-"princeton-nlp/SWE-bench_Lite"}
SPLIT=${8:-"test"}

echo ""
echo "=============================================================="
echo "Step 2: Running evaluation"
echo "=============================================================="

for OUTPUT_FILE in $OUTPUT_FILES; do
    if [ ! -f "$OUTPUT_FILE" ]; then
        echo "WARNING: $OUTPUT_FILE does not exist, skipping."
        continue
    fi
    echo ""
    echo "Evaluating: $OUTPUT_FILE"
    echo "--------------------------------------------------------------"
    bash "$SCRIPT_DIR/eval_infer.sh" "$OUTPUT_FILE" "" "$DATASET" "$SPLIT"
done

echo ""
echo "=============================================================="
echo "All done: inference + evaluation complete."
echo "=============================================================="
