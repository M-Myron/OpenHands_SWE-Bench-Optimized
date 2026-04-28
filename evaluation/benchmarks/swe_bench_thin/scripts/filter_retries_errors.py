#!/usr/bin/env python3
"""Filter maximum_retries_exceeded.jsonl for reprocessing, cross-referencing output.jsonl.

The retries file accumulates across runs. An instance may have failed in run 1
but succeeded in run 2 (and be present in output.jsonl). This script handles that:

  - If instance is already in output.jsonl → KEEP (harmless, already done)
  - If NOT in output.jsonl AND error is retriable infra → REMOVE (will be retried)
  - If NOT in output.jsonl AND error is non-retriable → KEEP (genuine failure)

Usage:
    python filter_retries_errors.py <maximum_retries_exceeded.jsonl> <output.jsonl> [--dry-run]
"""

import argparse
import json
import os
import shutil
import sys

# Retriable infrastructure error patterns (same as filter_output_errors.py)
RETRIABLE_PATTERNS = [
    "STATUS$ERROR_LLM_SERVICE_UNAVAILABLE",
    "UnixHTTPConnectionPool(host='localhost', port=None): Read timed out.",
    "pull access denied",
    "No such image",
    "[Errno 104] Connection reset by peer",
    "Too many open files",
    "cannot access local variable 'runtime'",
    "mmr1115/openhands-runtime:",
    "docker', 'buildx', 'build'",
    "404 Client Error",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("retries_file", help="Path to maximum_retries_exceeded.jsonl")
    parser.add_argument("output_file", help="Path to the (already filtered) output.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't modify")
    parser.add_argument("--output", help="Write filtered result here (default: overwrite in-place)")
    args = parser.parse_args()

    if not os.path.isfile(args.retries_file):
        print(f"Error: {args.retries_file} not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.output_file):
        print(f"Error: {args.output_file} not found", file=sys.stderr)
        sys.exit(1)

    # Load instance_ids from output.jsonl (already processed successfully)
    done_ids = set()
    with open(args.output_file) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                iid = data.get("instance_id", "")
                if iid:
                    done_ids.add(iid)
            except json.JSONDecodeError:
                pass

    print(f"Instances in output.jsonl (done): {len(done_ids)}")
    print(f"Retriable patterns ({len(RETRIABLE_PATTERNS)}):")
    for i, p in enumerate(RETRIABLE_PATTERNS, 1):
        print(f"  {i}. {p}")
    print()

    kept_lines = []
    removed_lines = []
    stats = {
        "already_done_kept": 0,
        "not_done_retriable_removed": 0,
        "not_done_genuine_kept": 0,
    }
    removed_by_pattern = {p: 0 for p in RETRIABLE_PATTERNS}
    total = 0

    with open(args.retries_file) as f:
        for line in f:
            total += 1
            stripped = line.strip()
            if not stripped:
                kept_lines.append(line)
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                kept_lines.append(line)
                continue

            iid = data.get("instance_id", "")
            error_val = str(data.get("error", "") or "")

            if iid in done_ids:
                # Instance already succeeded in output.jsonl — keep (harmless)
                kept_lines.append(line)
                stats["already_done_kept"] += 1
                continue

            # Instance NOT in output.jsonl — check if error is retriable
            matched_pattern = None
            for pattern in RETRIABLE_PATTERNS:
                if pattern in error_val:
                    matched_pattern = pattern
                    break

            if matched_pattern:
                # Retriable infra error, not in output → remove so it retries
                removed_lines.append(iid)
                removed_by_pattern[matched_pattern] += 1
                stats["not_done_retriable_removed"] += 1
            else:
                # Genuine failure, not in output → keep
                kept_lines.append(line)
                stats["not_done_genuine_kept"] += 1

    print(f"Total retries entries:  {total}")
    print(f"Kept entries:           {len(kept_lines)}")
    print(f"Removed entries:        {len(removed_lines)}")
    print()
    print(f"Breakdown:")
    print(f"  Already in output.jsonl (kept):     {stats['already_done_kept']}")
    print(f"  Retriable infra error (removed):    {stats['not_done_retriable_removed']}")
    print(f"  Genuine failure (kept):             {stats['not_done_genuine_kept']}")
    print()
    print("Removed by pattern:")
    for p, count in removed_by_pattern.items():
        if count > 0:
            print(f"  [{count:4d}] {p}")

    if args.dry_run:
        print("\n(dry-run mode — no files modified)")
        if removed_lines:
            print(f"\nFirst 30 removed instance_ids:")
            for iid in removed_lines[:30]:
                print(f"  {iid}")
        return

    if not removed_lines:
        print("\nNothing to filter. File unchanged.")
        return

    output_path = args.output or args.retries_file
    if output_path == args.retries_file:
        backup_path = args.retries_file + ".bak"
        shutil.copy2(args.retries_file, backup_path)
        print(f"\nBackup saved to: {backup_path}")

    with open(output_path, "w") as f:
        for line in kept_lines:
            f.write(line)

    print(f"Filtered output written to: {output_path}")


if __name__ == "__main__":
    main()
