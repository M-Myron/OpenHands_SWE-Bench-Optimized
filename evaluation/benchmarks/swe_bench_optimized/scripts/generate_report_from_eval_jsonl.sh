#!/usr/bin/env bash
# Generate report.json from output.swebench_eval.jsonl (alternative eval layout
# that does NOT use a per-instance eval_outputs/<instance>/report.json tree).
#
# Usage:
#   ./generate_report_from_eval_jsonl.sh <output_dir> [dataset_name] [split]
# ./evaluation/benchmarks/swe_bench_optimized/scripts/generate_report_from_eval_jsonl.sh /home/v-murongma/code/OpenHands_SWE-Bench-Optimized/evaluation/evaluation_outputs/outputs_thin/princeton-nlp__SWE-bench_Lite-test/CodeActAgent/policy_traj_128k_swegym_all_2092i_qwen2.5_coder_32b_full_128k_megatron_20260416_113234_maxiter_100_N_v0.61.0-thin-no-hint "princeton-nlp/SWE-bench_Lite" "test"

set -euo pipefail

OUTPUT_DIR="${1:?Usage: $0 <output_dir> [dataset_name] [split]}"
DATASET_NAME="${2:-}"
SPLIT="${3:-test}"

OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

EVAL_JSONL=""
for f in "$OUTPUT_DIR/output.swebench_eval.jsonl" "$OUTPUT_DIR/output.swebench.jsonl"; do
    if [ -f "$f" ]; then
        EVAL_JSONL="$f"
        break
    fi
done
if [ -z "$EVAL_JSONL" ]; then
    echo "Error: no output.swebench_eval.jsonl (or .swebench.jsonl) in $OUTPUT_DIR" >&2
    exit 1
fi

PREDICTIONS_FILE=""
if [ -f "$OUTPUT_DIR/output.jsonl" ]; then
    PREDICTIONS_FILE="$OUTPUT_DIR/output.jsonl"
fi

if [ -f "$OUTPUT_DIR/report.json" ]; then
    echo "Backing up existing report.json -> report.json.bak"
    cp "$OUTPUT_DIR/report.json" "$OUTPUT_DIR/report.json.bak"
fi

export OUTPUT_DIR EVAL_JSONL PREDICTIONS_FILE DATASET_NAME SPLIT
python3 << 'PYEOF'
import json
import os
import sys

output_dir = os.environ["OUTPUT_DIR"]
eval_jsonl = os.environ["EVAL_JSONL"]
predictions_file = os.environ.get("PREDICTIONS_FILE", "")
dataset_name = os.environ.get("DATASET_NAME", "")
split = os.environ.get("SPLIT", "test")

# 1) Parse eval jsonl into per-instance report
per_instance = {}
with open(eval_jsonl) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"  Warning: skipping bad line: {e}", file=sys.stderr)
            continue
        iid = d.get("instance_id")
        if not iid:
            continue
        tr = d.get("test_result", {}) or {}
        rep = tr.get("report", {}) or {}
        per_instance[iid] = {
            "resolved": bool(rep.get("resolved", False)),
            "patch_is_None": bool(rep.get("empty_generation", False)),
            "failed_apply_patch": bool(rep.get("failed_apply_patch", False)),
            "error_eval": bool(rep.get("error_eval", False)),
            "test_timeout": bool(rep.get("test_timeout", False)),
        }

completed_ids = sorted(per_instance.keys())
resolved_ids = sorted(k for k, v in per_instance.items() if v["resolved"])
unresolved_ids = sorted(k for k, v in per_instance.items() if not v["resolved"])
empty_patch_ids = sorted(k for k, v in per_instance.items() if v["patch_is_None"])

# 2) Determine total/submitted via predictions file or dataset
all_prediction_ids = set()
submitted_ids = set()
incomplete_ids = []
error_ids = []

if predictions_file and os.path.exists(predictions_file):
    with open(predictions_file) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = d.get("instance_id", "")
            if not iid:
                continue
            all_prediction_ids.add(iid)
            patch = (d.get("test_result", {}) or {}).get("git_patch", "") or d.get("model_patch", "")
            if patch and patch.strip():
                submitted_ids.add(iid)
    error_ids = sorted(submitted_ids - set(completed_ids))
    incomplete_ids = sorted(all_prediction_ids - submitted_ids)
    total_instances = len(all_prediction_ids)
    submitted_count = len(submitted_ids)
elif dataset_name:
    try:
        from datasets import load_dataset
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
if resolved_ids and total_instances:
    print(f"\n  Resolve rate: {len(resolved_ids)}/{total_instances} = "
          f"{100.0 * len(resolved_ids) / total_instances:.1f}%")
PYEOF

echo ""
echo "Done."
