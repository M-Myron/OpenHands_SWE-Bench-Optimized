"""Filter fact graphs by complexity.

Scans all ``stage2_facts.json`` files under a preprocess directory,
computes per-instance statistics, and writes a JSON filter file
listing instances that exceed the 95th-percentile thresholds for
non-problem-statement root count or total node count.

Usage:
    python -m evaluation.benchmarks.swe_bench_optimized.preprocess.filter_fact_graphs \
        --preprocess-dir <DIR> \
        --output <FILE>

Output JSON structure:
    {
        "thresholds": {
            "non_ps_roots_p95": 2.0,
            "total_nodes_p95": 50.0,
            "percentile": 95
        },
        "stats": {
            "total_instances": 924,
            "filtered_count": 43,
            "remaining_count": 881
        },
        "filtered_instance_ids": ["instance_a", ...],
        "remaining_instance_ids": ["instance_b", ...],
        "per_instance": [
            {
                "instance_id": "...",
                "total_nodes": 44,
                "total_roots": 5,
                "ps_roots": 1,
                "non_ps_roots": 4,
                "filtered": true,
                "filter_reason": "non_ps_roots=4 > 2.0"
            },
            ...
        ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np


def _classify_roots(nodes: list[dict]) -> dict[str, int]:
    """Return root-type counts for a fact graph."""
    ps_roots = 0
    non_ps_roots = 0
    for node in nodes:
        if node.get('depends_on'):
            continue
        unlocker = node.get('unlocker', {})
        action = unlocker.get('action', '') if isinstance(unlocker, dict) else ''
        if 'problem_statement' in action.lower():
            ps_roots += 1
        else:
            non_ps_roots += 1
    return {
        'total_roots': ps_roots + non_ps_roots,
        'ps_roots': ps_roots,
        'non_ps_roots': non_ps_roots,
    }


def compute_filter(
    preprocess_dir: str,
    percentile: float = 95,
) -> dict[str, Any]:
    """Scan preprocess dir and compute the filter list."""
    per_instance: list[dict] = []

    for iid in sorted(os.listdir(preprocess_dir)):
        facts_path = os.path.join(preprocess_dir, iid, 'stage2_facts.json')
        if not os.path.isfile(facts_path):
            continue
        with open(facts_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        nodes = data.get('nodes', [])
        rc = _classify_roots(nodes)
        per_instance.append({
            'instance_id': iid,
            'total_nodes': len(nodes),
            **rc,
        })

    if not per_instance:
        return {
            'thresholds': {},
            'stats': {'total_instances': 0, 'filtered_count': 0, 'remaining_count': 0},
            'filtered_instance_ids': [],
            'remaining_instance_ids': [],
            'per_instance': [],
        }

    non_ps_arr = np.array([r['non_ps_roots'] for r in per_instance])
    total_nodes_arr = np.array([r['total_nodes'] for r in per_instance])

    p_non_ps = float(np.percentile(non_ps_arr, percentile))
    p_total_nodes = float(np.percentile(total_nodes_arr, percentile))

    filtered_ids: list[str] = []
    remaining_ids: list[str] = []

    for rec in per_instance:
        reasons: list[str] = []
        if rec['non_ps_roots'] > p_non_ps:
            reasons.append(f"non_ps_roots={rec['non_ps_roots']} > {p_non_ps}")
        if rec['total_nodes'] > p_total_nodes:
            reasons.append(f"total_nodes={rec['total_nodes']} > {p_total_nodes}")
        rec['filtered'] = bool(reasons)
        rec['filter_reason'] = '; '.join(reasons) if reasons else ''
        if rec['filtered']:
            filtered_ids.append(rec['instance_id'])
        else:
            remaining_ids.append(rec['instance_id'])

    return {
        'thresholds': {
            'non_ps_roots_p95': p_non_ps,
            'total_nodes_p95': p_total_nodes,
            'percentile': percentile,
        },
        'stats': {
            'total_instances': len(per_instance),
            'filtered_count': len(filtered_ids),
            'remaining_count': len(remaining_ids),
        },
        'filtered_instance_ids': filtered_ids,
        'remaining_instance_ids': remaining_ids,
        'per_instance': per_instance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Filter fact graphs by root complexity')
    parser.add_argument(
        '--preprocess-dir',
        required=True,
        help='Path to swegym_v6 preprocess directory',
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to write the filter JSON',
    )
    parser.add_argument(
        '--percentile',
        type=float,
        default=95,
        help='Percentile threshold (default: 95)',
    )
    args = parser.parse_args()

    if not os.path.isdir(args.preprocess_dir):
        print(f'ERROR: Directory not found: {args.preprocess_dir}', file=sys.stderr)
        sys.exit(1)

    result = compute_filter(args.preprocess_dir, args.percentile)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print(f"Thresholds: non_ps_roots > {result['thresholds']['non_ps_roots_p95']}, "
          f"total_nodes > {result['thresholds']['total_nodes_p95']}")
    print(f"Total: {result['stats']['total_instances']}, "
          f"Filtered: {result['stats']['filtered_count']}, "
          f"Remaining: {result['stats']['remaining_count']}")
    print(f'Written to: {args.output}')


if __name__ == '__main__':
    main()
