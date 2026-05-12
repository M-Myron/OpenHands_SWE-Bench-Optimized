#!/usr/bin/env bash
set -eo pipefail

# Combined script: run thin-docker inference then automatically evaluate.
#
# Usage:
#   ./run_infer_and_eval.sh <MODEL_CONFIG> <COMMIT_HASH> <AGENT> <EVAL_LIMIT> \
#       <MAX_ITER> <NUM_WORKERS> <DATASET> <SPLIT> [MODE]
#   ./evaluation/benchmarks/swe_bench_thin/scripts/run_infer_and_eval.sh llm.eval_policy_traj_128k_swegym_all_2092i_qwen2-5_coder_32b_full_128k_megatron HEAD CodeActAgent 500 100 16 princeton-nlp/SWE-bench_Verified test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Guard against stuck relaunches
export OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS=${OH_RUNTIME_PREPARE_WAIT_LOG_SECONDS:-30}
export OH_RUNTIME_PREPARE_TIMEOUT_SECONDS=${OH_RUNTIME_PREPARE_TIMEOUT_SECONDS:-0}
export OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS=${OH_RUNTIME_PREPARE_STALE_LOCK_SECONDS:-7200}
export OH_RUNTIME_PREPARE_MAX_CONCURRENCY=${OH_RUNTIME_PREPARE_MAX_CONCURRENCY:-8}
# Gentler / longer waits to let containerd settle
export OH_THIN_DOCKER_PULL_RETRIES=8
export OH_THIN_DOCKER_PULL_BACKOFF_BASE=10
export OH_THIN_DOCKER_PULL_BACKOFF_MULTIPLIER=2
export OH_THIN_DOCKER_PULL_BACKOFF_MAX=240
# → waits: 10, 20, 40, 80, 160, 240, 240, 240 s

# Remove Docker images after each eval instance completes (prevents disk full).
# Set to "false" to keep images cached for faster re-runs.
# export EVAL_CLEANUP_IMAGES=false
export EVAL_CLEANUP_IMAGES=${EVAL_CLEANUP_IMAGES:-true}

# ── Step 1: Run thin inference ──
INFER_LOG=$(mktemp /tmp/thin_infer_log.XXXXXX)
trap "rm -f $INFER_LOG" EXIT

echo "=============================================================="
echo "Step 1: Running inference (thin docker)"
echo "=============================================================="
bash "$SCRIPT_DIR/run_infer.sh" "$@" 2>&1 | tee "$INFER_LOG"
INFER_EXIT=${PIPESTATUS[0]}

if [ "$INFER_EXIT" -ne 0 ]; then
    echo "ERROR: Inference failed with exit code $INFER_EXIT"
    exit $INFER_EXIT
fi

# ── Step 2: Extract output file path(s) from inference log ──
OUTPUT_FILES=$(grep -oP '### OUTPUT FILE(?:\s+FOR RUN \d+)?: \K[^\s]+' "$INFER_LOG" | sort -u)

if [ -z "$OUTPUT_FILES" ]; then
    echo "WARNING: Could not find OUTPUT FILE path in log. Searching for output.jsonl..."
    EVAL_OUTPUT_DIR=${10:-${EVAL_OUTPUT_DIR:-"evaluation/evaluation_outputs/outputs_thin"}}
    DATASET=${7:-"princeton-nlp/SWE-bench_Verified"}
    OUTPUT_FILES=$(find "$EVAL_OUTPUT_DIR" -path "*${DATASET##*/}*" -name "output.jsonl" -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -z "$OUTPUT_FILES" ]; then
        echo "ERROR: Could not locate any output.jsonl. Aborting evaluation."
        exit 1
    fi
fi

# ── Step 3: Run evaluation for each output file ──
DATASET=${7:-"princeton-nlp/SWE-bench_Verified"}
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
echo "All done: thin inference + evaluation complete."
echo "=============================================================="
