#!/usr/bin/env bash

# Evaluate inference results from SWE-Gym legacy format runs.
#
# This is a convenience wrapper around eval_infer.sh.
# The eval process itself is identical (it operates on output.jsonl files).
# Works with any dataset (SWE-bench, SWE-Gym, SWE-bench-Live, etc.)
#
# Usage (same args as eval_infer.sh):
#   bash evaluation/benchmarks/swe_bench_optimized/scripts/eval_infer_swegym_legacy.sh \
#       <output_file> [instance_id] [dataset_name] [split] [environment]

echo "=== Evaluating SWE-Gym Legacy Format Results ==="

# Delegate to the standard eval_infer.sh
exec bash "$(dirname "$0")/eval_infer.sh" "$@"
