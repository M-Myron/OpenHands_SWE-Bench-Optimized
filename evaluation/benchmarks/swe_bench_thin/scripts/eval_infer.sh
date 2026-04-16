#!/usr/bin/env bash
# eval_infer.sh for swe_bench_thin
# Delegates to the swe_bench_optimized eval_infer.sh since evaluation logic is identical.
# The thin runtime only affects inference, not evaluation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPTIMIZED_SCRIPT="$SCRIPT_DIR/../../swe_bench_optimized/scripts/eval_infer.sh"

if [ ! -f "$OPTIMIZED_SCRIPT" ]; then
    echo "ERROR: Cannot find eval_infer.sh at $OPTIMIZED_SCRIPT"
    exit 1
fi

exec bash "$OPTIMIZED_SCRIPT" "$@"
