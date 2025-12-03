#!/usr/bin/env bash

# Multi-run evaluation script for SWE-bench
# This script automatically evaluates multiple runs for the same task/model configuration
# Usage: ./eval_multirun.sh <base_dir> [dataset_name] [split] [environment]
#
# Examples:
#   ./eval_multirun.sh /path/to/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/gpt-4.1_maxiter_100_N_v0.61.0-no-hint
#   ./eval_multirun.sh /path/to/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/gpt-4.1_maxiter_100_N_v0.61.0-no-hint "princeton-nlp/SWE-bench_Verified" "test"

BASE_DIR=$1
if [ -z "$BASE_DIR" ]; then
    echo "Error: BASE_DIR is empty. Usage: ./eval_multirun.sh <base_dir> [dataset_name] [split] [environment]"
    echo ""
    echo "Examples:"
    echo "  ./eval_multirun.sh /path/to/outputs/CodeActAgent/gpt-4.1_maxiter_100_N_v0.61.0-no-hint"
    echo "  ./eval_multirun.sh /path/to/outputs/CodeActAgent/gpt-4.1_maxiter_100_N_v0.61.0-no-hint \"princeton-nlp/SWE-bench_Verified\" \"test\""
    exit 1
fi

DATASET_NAME=${2:-"princeton-nlp/SWE-bench_Verified"}
SPLIT=${3:-"test"}
ENVIRONMENT=${4:-"local"}

echo "=============================================================="
echo "Multi-run SWE-bench Evaluation Script"
echo "=============================================================="
echo "BASE_DIR: $BASE_DIR"
echo "DATASET_NAME: $DATASET_NAME"
echo "SPLIT: $SPLIT"
echo "ENVIRONMENT: $ENVIRONMENT"
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVAL_INFER_SCRIPT="$SCRIPT_DIR/eval_infer.sh"

if [ ! -f "$EVAL_INFER_SCRIPT" ]; then
    echo "Error: eval_infer.sh not found at $EVAL_INFER_SCRIPT"
    exit 1
fi

# Check if BASE_DIR contains a wildcard pattern
if [[ "$BASE_DIR" == *"*"* ]] || [[ "$BASE_DIR" == *"?"* ]]; then
    echo "Pattern detected in BASE_DIR: $BASE_DIR"
    # Find all matching directories
    MATCHING_DIRS=($(find "$(dirname "$BASE_DIR")" -maxdepth 1 -type d -name "$(basename "$BASE_DIR")" 2>/dev/null | sort))
    
    if [ ${#MATCHING_DIRS[@]} -eq 0 ]; then
        echo "Error: No directories found matching pattern: $BASE_DIR"
        exit 1
    fi
    
    echo "Found ${#MATCHING_DIRS[@]} directories matching pattern"
    echo ""
    
    RUN_DIRS=("${MATCHING_DIRS[@]}")
else
    # Check if it's a specific run directory (contains output.jsonl)
    if [ -f "$BASE_DIR/output.jsonl" ]; then
        echo "Single run directory detected"
        RUN_DIRS=("$BASE_DIR")
    else
        # Look for run subdirectories (run_1, run_2, etc.)
        RUN_DIRS=($(find "$BASE_DIR" -maxdepth 1 -type d -name "*run_*" 2>/dev/null | sort))
        
        if [ ${#RUN_DIRS[@]} -eq 0 ]; then
            # Try direct subdirectories with output.jsonl
            RUN_DIRS=($(find "$BASE_DIR" -maxdepth 1 -mindepth 1 -type d -exec test -f '{}/output.jsonl' \; -print 2>/dev/null | sort))
        fi
        
        if [ ${#RUN_DIRS[@]} -eq 0 ]; then
            echo "Error: No run directories found in $BASE_DIR"
            echo "Looking for directories containing output.jsonl"
            exit 1
        fi
        
        echo "Found ${#RUN_DIRS[@]} run directories"
        echo ""
    fi
fi

# Display found directories
echo "Run directories to evaluate:"
for i in "${!RUN_DIRS[@]}"; do
    echo "  [$((i+1))] ${RUN_DIRS[$i]}"
done
echo ""

# # Ask for confirmation
# read -p "Proceed with evaluation? (y/n) " -n 1 -r
# echo ""
# if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#     echo "Evaluation cancelled."
#     exit 0
# fi

# Evaluate each run
TOTAL_RUNS=${#RUN_DIRS[@]}
SUCCESSFUL_RUNS=0
FAILED_RUNS=0
FAILED_DIRS=()

echo ""
echo "=============================================================="
echo "Starting evaluation of $TOTAL_RUNS runs"
echo "=============================================================="
echo ""

for i in "${!RUN_DIRS[@]}"; do
    RUN_DIR="${RUN_DIRS[$i]}"
    RUN_NUM=$((i+1))
    
    echo "=============================================================="
    echo "[$RUN_NUM/$TOTAL_RUNS] Evaluating: $(basename "$RUN_DIR")"
    echo "=============================================================="
    
    OUTPUT_FILE="$RUN_DIR/output.jsonl"
    
    if [ ! -f "$OUTPUT_FILE" ]; then
        echo "Error: output.jsonl not found in $RUN_DIR"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        FAILED_DIRS+=("$RUN_DIR")
        echo ""
        continue
    fi
    
    # # Check if already evaluated
    # if [ -f "$RUN_DIR/report.json" ]; then
    #     read -p "Report already exists. Re-evaluate? (y/n) " -n 1 -r
    #     echo ""
    #     if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    #         echo "Skipping $(basename "$RUN_DIR")"
    #         echo ""
    #         continue
    #     fi
    # fi
    
    # Run evaluation
    START_TIME=$(date +%s)
    
    "$EVAL_INFER_SCRIPT" "$OUTPUT_FILE" "" "$DATASET_NAME" "$SPLIT" "$ENVIRONMENT"
    EXIT_CODE=$?
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Successfully evaluated in ${DURATION}s"
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
    else
        echo "✗ Evaluation failed with exit code $EXIT_CODE"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        FAILED_DIRS+=("$RUN_DIR")
    fi
    
    echo ""
done

# Summary
echo "=============================================================="
echo "Evaluation Summary"
echo "=============================================================="
echo "Total runs: $TOTAL_RUNS"
echo "Successful: $SUCCESSFUL_RUNS"
echo "Failed: $FAILED_RUNS"

if [ $FAILED_RUNS -gt 0 ]; then
    echo ""
    echo "Failed directories:"
    for dir in "${FAILED_DIRS[@]}"; do
        echo "  - $dir"
    done
fi

echo ""
echo "=============================================================="
echo "Multi-run evaluation complete!"
echo "=============================================================="

exit 0
