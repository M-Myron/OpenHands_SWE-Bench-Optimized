#!/usr/bin/env python3
"""Collect failed/empty-patch instances from output.jsonl and prepare for rerun.

This script reads an output.jsonl from a rollout, identifies instances that should
be rerun (empty patch, infrastructure errors, etc.), removes them from the output
files, and optionally triggers a rerun.

Usage:
    # 1. Analyze only (dry run) - see what would be rerun:
    python collect_and_rerun_failed.py --output-file /path/to/output.jsonl --dry-run

    # 2. Remove failed instances from output.jsonl so the next rollout picks them up:
    python collect_and_rerun_failed.py --output-file /path/to/output.jsonl

    # 3. Also include instances with errors (not just empty patches):
    python collect_and_rerun_failed.py --output-file /path/to/output.jsonl \
        --include-errors max_retries,runtime_error

Error categories (use with --include-errors):
    empty_patch       : Agent produced no git diff (always included by default)
    max_retries       : Infrastructure failures (docker build, cd failures, etc.)
    retryable_error   : Transient LLM errors (internal server error, service unavailable)
    all_errors        : Everything except max_iteration and stuck_in_loop
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime


def classify_instance(data: dict) -> dict:
    """Classify an instance into categories based on its output."""
    tr = data.get('test_result', {})
    patch = tr.get('git_patch', '')
    error = data.get('error')

    result = {
        'instance_id': data.get('instance_id', 'unknown'),
        'empty_patch': not patch or not patch.strip(),
        'has_error': bool(error),
        'error_category': None,
        'error_summary': None,
    }

    if error:
        err_str = str(error)
        if 'Agent reached maximum iteration' in err_str:
            result['error_category'] = 'max_iteration'
        elif 'AgentStuckInLoopError' in err_str:
            result['error_category'] = 'stuck_in_loop'
        elif 'Maximum retries' in err_str and (
            'docker' in err_str.lower()
            or 'buildx' in err_str.lower()
            or 'Failed to cd' in err_str
            or 'RuntimeError' in err_str
        ):
            result['error_category'] = 'max_retries_infrastructure'
        elif 'Maximum retries' in err_str and (
            'Retryable controller error' in err_str
            or 'STATUS$ERROR_LLM' in err_str
        ):
            result['error_category'] = 'max_retries_retryable'
        elif 'Maximum retries' in err_str:
            result['error_category'] = 'max_retries_other'
        elif 'EvalTimeoutException' in err_str or 'timed out' in err_str.lower():
            result['error_category'] = 'timeout'
        else:
            result['error_category'] = 'other_error'
        result['error_summary'] = err_str[:120]

    return result


def should_rerun(classification: dict, include_errors: set) -> bool:
    """Decide whether an instance should be rerun based on its classification."""
    # Always rerun empty patches (unless they have max_iteration or stuck_in_loop errors,
    # which means the agent legitimately ran but failed to produce a patch)
    if classification['empty_patch']:
        cat = classification['error_category']
        # If the agent ran to max iterations or got stuck, the empty patch is a
        # "legitimate" failure — only rerun if user explicitly wants all_errors
        if cat in ('max_iteration', 'stuck_in_loop'):
            return 'all_errors' in include_errors or 'max_iteration' in include_errors
        # Otherwise empty patch with no error or with infra error = rerun
        return True

    # Non-empty patch but with infrastructure/retryable errors → rerun
    cat = classification['error_category']
    if not cat:
        return False  # No error, has patch — this is fine

    if 'all_errors' in include_errors:
        return cat not in ('max_iteration', 'stuck_in_loop')

    if 'max_retries' in include_errors and cat in (
        'max_retries_infrastructure',
        'max_retries_retryable',
        'max_retries_other',
    ):
        return True

    if 'retryable_error' in include_errors and cat == 'max_retries_retryable':
        return True

    if 'runtime_error' in include_errors and cat == 'max_retries_infrastructure':
        return True

    if 'timeout' in include_errors and cat == 'timeout':
        return True

    return False


def load_output_file(filepath: str) -> list[tuple[str, dict | None]]:
    """Load output.jsonl returning (raw_line, parsed_data_or_None) tuples."""
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            raw = line.rstrip('\n')
            if not raw.strip():
                entries.append((raw, None))
                continue
            try:
                data = json.loads(raw)
                entries.append((raw, data))
            except json.JSONDecodeError:
                entries.append((raw, None))
    return entries


def main():
    parser = argparse.ArgumentParser(
        description='Collect failed instances and prepare for rerun.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Path to output.jsonl from rollout',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only analyze and print results, do not modify files',
    )
    parser.add_argument(
        '--include-errors',
        type=str,
        default='max_retries,retryable_error,runtime_error,timeout',
        help=(
            'Comma-separated error categories to include for rerun. '
            'Options: max_retries, retryable_error, runtime_error, timeout, all_errors, max_iteration. '
            'Default: max_retries,retryable_error,runtime_error,timeout'
        ),
    )
    parser.add_argument(
        '--save-ids',
        type=str,
        default=None,
        help='Save the list of rerun instance IDs to this file (one per line)',
    )

    args = parser.parse_args()

    output_file = args.output_file
    if not os.path.exists(output_file):
        print(f'Error: Output file not found: {output_file}', file=sys.stderr)
        sys.exit(1)

    include_errors = set(x.strip() for x in args.include_errors.split(',') if x.strip())

    # Load and classify all instances
    entries = load_output_file(output_file)
    classifications = []
    for _raw, data in entries:
        if data is not None:
            classifications.append(classify_instance(data))

    # Determine which to rerun
    rerun_ids = set()
    rerun_classifications = []
    for cls in classifications:
        if should_rerun(cls, include_errors):
            rerun_ids.add(cls['instance_id'])
            rerun_classifications.append(cls)

    # Print analysis
    total = len(classifications)
    empty_patch_count = sum(1 for c in classifications if c['empty_patch'])
    error_counts = {}
    for c in classifications:
        cat = c['error_category'] or 'no_error'
        error_counts[cat] = error_counts.get(cat, 0) + 1

    print('=' * 70)
    print('ANALYSIS OF OUTPUT FILE')
    print('=' * 70)
    print(f'Output file: {output_file}')
    print(f'Total instances: {total}')
    print(f'Empty patches: {empty_patch_count}')
    print()
    print('Error category breakdown:')
    for cat, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        print(f'  {cat}: {count}')
    print()
    print(f'Include error categories for rerun: {include_errors}')
    print(f'Instances to rerun: {len(rerun_ids)}')
    print()

    # Breakdown of rerun reasons
    rerun_reasons = {}
    for c in rerun_classifications:
        if c['empty_patch']:
            reason = f'empty_patch (error: {c["error_category"] or "none"})'
        else:
            reason = f'error: {c["error_category"]}'
        rerun_reasons[reason] = rerun_reasons.get(reason, 0) + 1

    print('Rerun breakdown:')
    for reason, count in sorted(rerun_reasons.items(), key=lambda x: -x[1]):
        print(f'  {reason}: {count}')
    print()

    if args.dry_run:
        print('[DRY RUN] No files modified.')
        if rerun_ids:
            print()
            print(f'Instance IDs to rerun ({len(rerun_ids)}):')
            for iid in sorted(rerun_ids):
                cls = next(c for c in rerun_classifications if c['instance_id'] == iid)
                reason = 'empty_patch' if cls['empty_patch'] else cls['error_category']
                print(f'  {iid} ({reason})')
        if args.save_ids:
            with open(args.save_ids, 'w') as f:
                for iid in sorted(rerun_ids):
                    f.write(iid + '\n')
            print(f'\nSaved {len(rerun_ids)} instance IDs to {args.save_ids}')
        return

    if not rerun_ids:
        print('No instances to rerun. Exiting.')
        return

    # Save instance IDs if requested
    if args.save_ids:
        with open(args.save_ids, 'w') as f:
            for iid in sorted(rerun_ids):
                f.write(iid + '\n')
        print(f'Saved {len(rerun_ids)} instance IDs to {args.save_ids}')

    # Backup and rewrite output.jsonl (remove rerun instances)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = output_file + f'.backup_{timestamp}'
    shutil.copy2(output_file, backup_file)
    print(f'Backed up {output_file} -> {backup_file}')

    kept = 0
    removed = 0
    with open(output_file, 'w') as f:
        for raw, data in entries:
            if data is None:
                # Skip empty/corrupted lines
                continue
            iid = str(data.get('instance_id', ''))
            if iid in rerun_ids:
                removed += 1
                continue
            f.write(raw + '\n')
            kept += 1

    print(f'Rewrote {output_file}: kept {kept}, removed {removed}')

    # Also clean up .swebench_eval.jsonl if it exists
    eval_file = output_file.replace('.jsonl', '.swebench_eval.jsonl')
    if os.path.exists(eval_file):
        eval_backup = eval_file + f'.backup_{timestamp}'
        shutil.copy2(eval_file, eval_backup)
        print(f'Backed up {eval_file} -> {eval_backup}')

        eval_entries = load_output_file(eval_file)
        eval_kept = 0
        eval_removed = 0
        with open(eval_file, 'w') as f:
            for raw, data in eval_entries:
                if data is None:
                    continue
                iid = str(data.get('instance_id', ''))
                if iid in rerun_ids:
                    eval_removed += 1
                    continue
                f.write(raw + '\n')
                eval_kept += 1
        print(f'Rewrote {eval_file}: kept {eval_kept}, removed {eval_removed}')

    print()
    print('=' * 70)
    print('DONE! Now rerun the rollout to process the removed instances.')
    print(f'Removed {removed} instances from output.jsonl.')
    print('The rollout script will automatically pick up the missing instances.')
    print('=' * 70)


if __name__ == '__main__':
    main()
