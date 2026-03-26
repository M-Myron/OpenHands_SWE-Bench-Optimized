#!/usr/bin/env python3
"""
Scan eval log files for instances where patch_successfully_applied=False,
remove them from the swebench_eval.jsonl resume file so they get reprocessed,
then re-run the eval command.

Usage:
    python reprocess_patch_apply_failed.py [--dry-run] [--input-file PATH]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_INPUT_FILE = (
    "/home/v-murongma/code/OpenHands_SWE-Bench-Optimized/evaluation/evaluation_outputs/"
    "outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/"
    "Qwen3-Coder-480B-A35B-Instruct_maxiter_100_N_v0.61.0-no-hint-train-qwen3_coder_480b_a35b_instruct-t05/"
    "Qwen3-Coder-480B-A35B-Instruct_maxiter_100_N_v0.61.0-no-hint-train-qwen3_coder_480b_a35b_instruct-t05-run_5/"
    "output.jsonl"
)

EVAL_CMD_TEMPLATE = (
    "poetry run python evaluation/benchmarks/swe_bench/eval_infer.py"
    " --eval-num-workers 4"
    " --input-file {input_file}"
    ' --dataset "SWE-Gym/SWE-Gym"'
    ' --split "train"'
)

# ─── Step 1: Scan logs for patch_successfully_applied=False ───────────────────

def find_problematic_instances(log_dir: Path) -> list[str]:
    """Return list of instance_ids where patch_successfully_applied is False."""
    problematic: list[str] = []

    if not log_dir.exists():
        print(f"[WARN] Log directory does not exist: {log_dir}")
        return problematic

    log_files = list(log_dir.glob("instance_*.log"))
    print(f"[INFO] Scanning {len(log_files)} log files in {log_dir} ...")

    # Pattern: '...report: {...'patch_successfully_applied': False...}'
    report_pattern = re.compile(
        r"\[([^\]]+)\] report: (\{.*\})"
    )

    for lf in sorted(log_files):
        instance_id_from_filename = lf.stem.replace("instance_", "", 1)
        found_problem = False
        with open(lf, "r", errors="replace") as fh:
            for line in fh:
                m = report_pattern.search(line)
                if m:
                    instance_id_in_log = m.group(1)
                    try:
                        report = json.loads(m.group(2).replace("'", '"'))
                        if report.get("patch_successfully_applied") is False:
                            found_problem = True
                            problematic.append(instance_id_in_log)
                            break
                    except json.JSONDecodeError:
                        # Fall back to simple string search
                        if "'patch_successfully_applied': False" in line or \
                           '"patch_successfully_applied": false' in line:
                            found_problem = True
                            problematic.append(instance_id_from_filename)
                            break

    print(f"[INFO] Found {len(problematic)} instance(s) with patch_successfully_applied=False")
    return problematic


# ─── Step 2: Remove problematic instances from the resume file ────────────────

def remove_from_resume_file(eval_jsonl: Path, instance_ids: set[str], dry_run: bool) -> int:
    """
    Remove lines whose instance_id is in `instance_ids` from the resume file.
    Returns the number of lines removed.
    """
    if not eval_jsonl.exists():
        print(f"[INFO] Resume file does not exist yet: {eval_jsonl} — nothing to remove.")
        return 0

    kept_lines: list[str] = []
    removed_count = 0

    with open(eval_jsonl, "r", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                kept_lines.append(line)
                continue
            try:
                data = json.loads(stripped)
                iid = str(data.get("instance_id", ""))
                if iid in instance_ids:
                    removed_count += 1
                    print(f"  [REMOVE] {iid}")
                    continue
            except json.JSONDecodeError:
                pass  # keep corrupted lines as-is; shared.py handles them
            kept_lines.append(line)

    if dry_run:
        print(f"[DRY-RUN] Would remove {removed_count} line(s) from {eval_jsonl}")
    else:
        with open(eval_jsonl, "w") as fh:
            fh.writelines(kept_lines)
        print(f"[INFO] Removed {removed_count} line(s) from {eval_jsonl}")

    return removed_count


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        default=DEFAULT_INPUT_FILE,
        help="Path to the output.jsonl (input to eval_infer.py)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without modifying files or running eval.",
    )
    args = parser.parse_args()

    input_file = Path(args.input_file)
    run_dir = input_file.parent

    # Derived paths (mirrors eval_infer.py logic)
    eval_jsonl = run_dir / (input_file.stem + ".swebench_eval.jsonl")
    log_dir    = run_dir / (input_file.stem + ".swebench_eval.logs")

    print("=" * 70)
    print(f"Input file   : {input_file}")
    print(f"Resume file  : {eval_jsonl}")
    print(f"Log directory: {log_dir}")
    print("=" * 70)

    # ── Step 1 ──
    problematic = find_problematic_instances(log_dir)
    if not problematic:
        print("[INFO] No problematic instances found. Nothing to reprocess.")
        sys.exit(0)

    print("\nProblematic instance IDs:")
    for iid in sorted(problematic):
        print(f"  {iid}")

    problematic_set = set(problematic)

    # ── Step 2 ──
    print()
    remove_from_resume_file(eval_jsonl, problematic_set, dry_run=args.dry_run)

    # ── Step 3: Re-run eval ──
    cmd = EVAL_CMD_TEMPLATE.format(input_file=str(input_file))
    print(f"\n[INFO] Running eval command:\n  {cmd}\n")

    if args.dry_run:
        print("[DRY-RUN] Skipping actual execution.")
        sys.exit(0)

    repo_root = Path(__file__).parent
    result = subprocess.run(cmd, shell=True, cwd=str(repo_root))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
