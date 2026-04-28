#!/usr/bin/env python3
"""Filter lines from output.jsonl whose 'error' field matches specified patterns.

Usage:
    python filter_output_errors.py <input.jsonl> [--output <output.jsonl>] [--dry-run]

If --output is not specified, the input file is overwritten in-place (a .bak backup is created).
Use --dry-run to only report counts without modifying anything.

Edit ERROR_PATTERNS below to add/remove patterns to filter out.
"""

import argparse
import json
import os
import shutil
import sys

# ============================================================================
# ERROR PATTERNS TO FILTER OUT
# Each entry is a substring match against the 'error' field of each JSONL line.
# If the error field *contains* any of these substrings, the line is removed.
# Add or comment out patterns as needed.
# ============================================================================
ERROR_PATTERNS = [
    # LLM service unavailable retries
    "STATUS$ERROR_LLM_SERVICE_UNAVAILABLE",

    # Docker read timeouts (daemon overload during cleanup)
    "UnixHTTPConnectionPool(host='localhost', port=None): Read timed out.",

    # Docker image not found (pull access denied — image doesn't exist on Hub)
    "pull access denied",

    # Docker image not found (not cached locally, pull failed)
    "No such image",

    # Connection reset (network/Docker daemon flake)
    "[Errno 104] Connection reset by peer",

    # Too many open files (fd exhaustion under high concurrency)
    "Too many open files",

    # Cannot access local variable 'runtime' (race condition on cleanup)
    "cannot access local variable 'runtime'",

    # Wrong runtime used (openhands-runtime overlay instead of thin_docker)
    "mmr1115/openhands-runtime:",

    # Docker buildx errors (wrong runtime tried to build overlay)
    "docker', 'buildx', 'build'",
]


def matches_any_pattern(error_value: str, patterns: list[str]) -> bool:
    """Return True if the error string contains any of the patterns."""
    if not error_value:
        return False
    for pattern in patterns:
        if pattern in error_value:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Filter lines from output.jsonl by error patterns."
    )
    parser.add_argument("input", help="Path to input JSONL file")
    parser.add_argument(
        "--output", "-o",
        help="Path to output JSONL file. If omitted, overwrites input (with .bak backup).",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Only report what would be filtered; do not write any files.",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    active_patterns = [p for p in ERROR_PATTERNS if p]
    if not active_patterns:
        print("No active error patterns defined. Edit ERROR_PATTERNS in the script.")
        sys.exit(0)

    print(f"Active error patterns ({len(active_patterns)}):")
    for i, p in enumerate(active_patterns, 1):
        print(f"  {i}. {p}")
    print()

    kept_lines = []
    removed_lines = []
    removed_by_pattern: dict[str, int] = {p: 0 for p in active_patterns}
    total = 0

    with open(input_path, "r") as f:
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

            error_val = str(data.get("error", "") or "")
            matched = False
            for pattern in active_patterns:
                if pattern in error_val:
                    removed_by_pattern[pattern] += 1
                    matched = True
                    break  # count each line only once
            if matched:
                removed_lines.append(data.get("instance_id", "???"))
            else:
                kept_lines.append(line)

    print(f"Total lines:   {total}")
    print(f"Kept lines:    {len(kept_lines)}")
    print(f"Removed lines: {len(removed_lines)}")
    print()
    print("Removed by pattern:")
    for p, count in removed_by_pattern.items():
        print(f"  [{count:4d}] {p}")

    if args.dry_run:
        print("\n(dry-run mode — no files modified)")
        if removed_lines:
            print(f"\nFirst 20 removed instance_ids:")
            for iid in removed_lines[:20]:
                print(f"  {iid}")
        return

    if not removed_lines:
        print("\nNothing to filter. File unchanged.")
        return

    # Determine output path
    output_path = args.output or input_path
    if output_path == input_path:
        backup_path = input_path + ".bak"
        shutil.copy2(input_path, backup_path)
        print(f"\nBackup saved to: {backup_path}")

    with open(output_path, "w") as f:
        for line in kept_lines:
            f.write(line)

    print(f"Filtered output written to: {output_path}")


if __name__ == "__main__":
    main()
