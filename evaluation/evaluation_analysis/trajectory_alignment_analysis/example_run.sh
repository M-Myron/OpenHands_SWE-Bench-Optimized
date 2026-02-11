#!/bin/bash

# Example script showing how to run the judge evaluator
# This demonstrates the typical workflow

set -e  # Exit on error

# Configuration
INPUT_FILE="/home/v-murongma/code/OpenHands_SWE-Bench-Optimized/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/gpt-4.1_maxiter_100_N_v0.61.0-no-hint/gpt-4.1_maxiter_100_N_v0.61.0-no-hint-run_1/output.with_completions.jsonl.gz"
OUTPUT_DIR="./judge_evaluation_results_$(date +%Y%m%d_%H%M%S)"
MODEL="gpt-4.1"

echo "================================================================"
echo "SWE-Bench Judge Evaluation - Example Run"
echo "================================================================"
echo ""
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL"
echo ""

# Step 1: Run evaluation on a small sample (10 instances)
echo "Step 1: Running evaluation on 10 instances (test run)..."
python judge_evaluator.py \
  --input-files "$INPUT_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --limit 100 \
  --max-retries 5 \
  --checkpoint-interval 5 \
  --max-workers 4 \
  --save-prompts

echo ""
echo "Step 1 complete!"
echo ""

# Step 2: Check the summary
echo "Step 2: Checking evaluation summary..."
echo ""
cat "$OUTPUT_DIR/evaluation_summary.json"
echo ""

# Step 3: Run analysis
echo "Step 3: Running result analysis..."
echo ""
python analyze_results.py \
  --results "$OUTPUT_DIR/evaluation_results.jsonl"

echo ""

# Step 4: Export to CSV
echo "Step 4: Exporting to CSV for further analysis..."
python analyze_results.py \
  --results "$OUTPUT_DIR/evaluation_results.jsonl" \
  --export-csv "$OUTPUT_DIR/analysis.csv"

echo ""
echo "================================================================"
echo "Example run complete!"
echo "================================================================"
echo ""
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Files generated:"
echo "  - evaluation_results.jsonl  : Full evaluation results"
echo "  - checkpoint.jsonl          : Checkpoint for resuming"
echo "  - evaluation_summary.json   : Summary statistics"
echo "  - analysis.csv              : Flattened data for analysis"
echo ""
echo "To run full evaluation (all instances), remove --limit flag:"
echo ""
echo "  python judge_evaluator.py \\"
echo "    --input-files \"$INPUT_FILE\" \\"
echo "    --output-dir \"$OUTPUT_DIR\" \\"
echo "    --model \"$MODEL\""
echo ""
echo "To resume an interrupted run, just re-run the same command."
echo ""
