#!/usr/bin/env bash
# Generate (or regenerate) a report.json from existing eval_outputs/ directory.
#
# Usage:
#   ./generate_report.sh <output_dir>
#   ./generate_report.sh <output_dir> [dataset_name] [split]
#
# Examples:
#   ./generate_report.sh evaluation/evaluation_outputs/outputs/.../my_run/
#   ./generate_report.sh evaluation/evaluation_outputs/outputs/.../my_run/ "princeton-nlp/SWE-bench_Verified" "test"
#
# This reads all per-instance report.json files inside <output_dir>/eval_outputs/
# and produces an aggregated report.json in <output_dir>/report.json.
# It also regenerates the README.md summary.
# ./evaluation/benchmarks/swe_bench_optimized/scripts/generate_report.sh /home/v-murongma/code/OpenHands_SWE-Bench-Optimized/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/policy_traj_128k_swegym_all_2092i_qwen2.5_coder_32b_full_128k_megatron_20260416_113234_maxiter_100_N_v0.61.0-no-hint "princeton-nlp/SWE-bench_Verified" "test"
# ./evaluation/benchmarks/swe_bench_optimized/scripts/generate_report.sh /home/v-murongma/code/OpenHands_SWE-Bench-Optimized/evaluation/evaluation_outputs/outputs_thin/princeton-nlp__SWE-bench_Lite-test/CodeActAgent/policy_traj_128k_swegym_resolved_752i_qwen2.5_coder_32b_full_128k_megatron_20260428_155440_maxiter_100_N_v0.61.0-thin-no-hint "princeton-nlp/SWE-bench_Lite" "test"


set -euo pipefail

OUTPUT_DIR="${1:?Usage: $0 <output_dir> [dataset_name] [split]}"
DATASET_NAME="${2:-}"
SPLIT="${3:-test}"

# Resolve to absolute path
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

EVAL_OUTPUTS_DIR="$OUTPUT_DIR/eval_outputs"
if [ ! -d "$EVAL_OUTPUTS_DIR" ]; then
    echo "Error: $EVAL_OUTPUTS_DIR does not exist."
    exit 1
fi

INSTANCE_COUNT=$(find "$EVAL_OUTPUTS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
REPORT_COUNT=$(find "$EVAL_OUTPUTS_DIR" -mindepth 2 -maxdepth 2 -name "report.json" | wc -l)
echo "Found $INSTANCE_COUNT instance directories, $REPORT_COUNT with report.json"

if [ "$REPORT_COUNT" -eq 0 ]; then
    echo "Error: No per-instance report.json files found in $EVAL_OUTPUTS_DIR"
    exit 1
fi

# Try to find the predictions file (output.jsonl or .swebench.jsonl)
PREDICTIONS_FILE=""
for f in "$OUTPUT_DIR/output.swebench.jsonl" "$OUTPUT_DIR/output.jsonl"; do
    if [ -f "$f" ]; then
        PREDICTIONS_FILE="$f"
        break
    fi
done

# Back up existing report
if [ -f "$OUTPUT_DIR/report.json" ]; then
    echo "Backing up existing report.json -> report.json.bak"
    cp "$OUTPUT_DIR/report.json" "$OUTPUT_DIR/report.json.bak"
fi

# ------------------------------------------------------------------
# Generate report from eval_outputs
# ------------------------------------------------------------------
export OUTPUT_DIR PREDICTIONS_FILE DATASET_NAME SPLIT
python3 << 'PYEOF'
import json
import glob
import os
import sys

output_dir = os.environ["OUTPUT_DIR"]
eval_outputs_dir = os.path.join(output_dir, "eval_outputs")
predictions_file = os.environ.get("PREDICTIONS_FILE", "")
dataset_name = os.environ.get("DATASET_NAME", "")

# 1) Collect per-instance reports
per_instance = {}
report_files = sorted(glob.glob(os.path.join(eval_outputs_dir, "*", "report.json")))
for rpath in report_files:
    try:
        with open(rpath) as f:
            data = json.load(f)
        per_instance.update(data)
    except (json.JSONDecodeError, IOError) as e:
        instance_id = os.path.basename(os.path.dirname(rpath))
        print(f"  Warning: could not read {rpath}: {e}", file=sys.stderr)

completed_ids = sorted(per_instance.keys())
resolved_ids = sorted(k for k, v in per_instance.items() if v.get("resolved", False))
unresolved_ids = sorted(k for k, v in per_instance.items() if not v.get("resolved", False))
empty_patch_ids = sorted(k for k, v in per_instance.items() if v.get("patch_is_None", False))

# 2) Determine total/submitted from predictions file if available
all_prediction_ids = set()
submitted_ids = set()
incomplete_ids = []
error_ids = []

if predictions_file and os.path.exists(predictions_file):
    with open(predictions_file) as f:
        for line in f:
            try:
                d = json.loads(line)
                iid = d.get("instance_id", "")
                if not iid:
                    continue
                all_prediction_ids.add(iid)
                patch = d.get("model_patch", "")
                if patch and patch.strip():
                    submitted_ids.add(iid)
            except json.JSONDecodeError:
                continue

    # Error instances: submitted but no report.json
    error_ids = sorted(submitted_ids - set(completed_ids))
    incomplete_ids = sorted(all_prediction_ids - submitted_ids)
    total_instances = len(all_prediction_ids)
    submitted_count = len(submitted_ids)
elif dataset_name:
    # Try to get total from dataset
    try:
        from datasets import load_dataset
        split = os.environ.get("SPLIT", "test")
        ds = load_dataset(dataset_name, split=split)
        total_instances = len(ds)
        all_prediction_ids = set(row["instance_id"] for row in ds)
        error_ids = sorted(all_prediction_ids - set(completed_ids))
    except Exception as e:
        print(f"  Warning: could not load dataset {dataset_name}: {e}", file=sys.stderr)
        total_instances = len(completed_ids)
    submitted_count = len(completed_ids)
else:
    total_instances = len(completed_ids)
    submitted_count = len(completed_ids)

# 3) Build report
report = {
    "total_instances": total_instances,
    "submitted_instances": submitted_count,
    "completed_instances": len(completed_ids),
    "resolved_instances": len(resolved_ids),
    "unresolved_instances": len(unresolved_ids),
    "empty_patch_instances": len(empty_patch_ids),
    "error_instances": len(error_ids),
    "schema_version": 2,
    "completed_ids": completed_ids,
    "resolved_ids": resolved_ids,
    "unresolved_ids": unresolved_ids,
    "empty_patch_ids": empty_patch_ids,
    "error_ids": error_ids,
    "incomplete_ids": incomplete_ids,
    "submitted_ids": sorted(submitted_ids) if submitted_ids else completed_ids,
}
# Add per-instance details
report.update(per_instance)

report_path = os.path.join(output_dir, "report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=4)

print(f"\nReport written to {report_path}")
print(f"  Total instances:     {report['total_instances']}")
print(f"  Submitted:           {report['submitted_instances']}")
print(f"  Completed (evaled):  {report['completed_instances']}")
print(f"  Resolved:            {report['resolved_instances']}")
print(f"  Unresolved:          {report['unresolved_instances']}")
print(f"  Empty patch:         {report['empty_patch_instances']}")
print(f"  Errors (not evaled): {report['error_instances']}")
if resolved_ids:
    pct = 100.0 * len(resolved_ids) / total_instances
    print(f"\n  Resolve rate: {len(resolved_ids)}/{total_instances} = {pct:.1f}%")
PYEOF

# ------------------------------------------------------------------
# Regenerate README.md via update_output_with_eval.py if output.jsonl exists
# ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPDATE_SCRIPT="$SCRIPT_DIR/../../swe_bench/scripts/eval/update_output_with_eval.py"

if [ -f "$OUTPUT_DIR/output.jsonl" ] && [ -f "$UPDATE_SCRIPT" ]; then
    echo ""
    echo "Regenerating README.md..."
    poetry run python "$UPDATE_SCRIPT" "$OUTPUT_DIR/output.jsonl" 2>/dev/null || true
fi

echo ""
echo "Done."
