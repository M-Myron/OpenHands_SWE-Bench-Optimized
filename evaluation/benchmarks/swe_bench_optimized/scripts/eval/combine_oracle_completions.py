"""Build raw_completions for oracle-guided runs.

Unlike combine_final_completions.py which reads the last LLM completion file
(which only has the *solver's* response), this script reconstructs the correct
SFT data by using:

1. The last LLM completion file's **messages** — these already contain oracle-
   revised content for all previous turns (the revised action is what gets
   executed, so the next solver call sees the oracle's version in its context).

2. The **model_response** from output.jsonl's last agent action's
   tool_call_metadata — this is the fn-call format response that was actually
   executed (oracle-guided or solver, depending on the planner's decision).

This ensures the SFT data captures the oracle's revisions including finish
calls, rather than the solver's rejected proposals.

Usage:
    python combine_oracle_completions.py /path/to/output.jsonl
"""

import argparse
import gzip
import json
import os
from glob import glob

from tqdm import tqdm


def load_last_completion(output_dir: str, instance_id: str):
    """Load messages and tools from the last LLM completion file."""
    glob_path = os.path.join(output_dir, 'llm_completions', instance_id, '*.json')
    files = sorted(glob(glob_path))
    if not files:
        return None
    with open(files[-1], 'r') as f:
        result = json.load(f)
    return {
        'messages': result['messages'],
        'tools': result['kwargs'].get('tools', []),
    }


def get_last_model_response(history: list[dict]):
    """Extract the model_response from the last agent action in history.

    Returns the fn-call format response dict (with content + tool_calls)
    that was actually executed, whether it came from the solver or the oracle.
    """
    for h in reversed(history):
        if h.get('source') != 'agent':
            continue
        tcm = h.get('tool_call_metadata') or h.get('args', {}).get('tool_call_metadata')
        if not tcm:
            # MessageAction (AWAITING_USER_INPUT) — no model_response
            if h.get('action') == 'message':
                continue
            continue
        mr = tcm.get('model_response')
        if mr and mr.get('choices'):
            msg = mr['choices'][0]['message']
            return msg
    return None


def build_raw_completions(
    completion_data: dict,
    last_response: dict | None,
) -> dict | None:
    """Build raw_completions dict with correct messages and tools.

    Args:
        completion_data: {messages, tools} from the last completion file
        last_response: The model_response message from output.jsonl's last
                      agent action (fn-call format with content + tool_calls)

    Returns:
        {messages: [...], tools: [...]} where messages includes all turns
        plus the correct last response.
    """
    messages = list(completion_data['messages'])

    if last_response:
        # Build the response message in the same format as completion files
        resp_msg = {
            'role': last_response.get('role', 'assistant'),
        }
        if last_response.get('content') is not None:
            resp_msg['content'] = last_response['content']
        if last_response.get('tool_calls'):
            resp_msg['tool_calls'] = last_response['tool_calls']

        messages.append(resp_msg)

    return {
        'messages': messages,
        'tools': completion_data.get('tools', []),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Build oracle-corrected raw_completions for SFT data.'
    )
    parser.add_argument('jsonl_path', type=str, help='Path to output.jsonl')
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output path (default: output.with_oracle_completions.jsonl.gz)',
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.jsonl_path)
    output_path = args.output or os.path.join(
        output_dir, 'output.with_oracle_completions.jsonl.gz'
    )

    stats = {
        'total': 0,
        'oracle_response': 0,
        'solver_response': 0,
        'no_completion': 0,
        'no_response': 0,
    }

    with (
        open(args.jsonl_path, 'r') as f_in,
        gzip.open(output_path, 'wt') as f_out,
    ):
        for line in tqdm(f_in, desc='Processing'):
            data = json.loads(line)
            stats['total'] += 1
            instance_id = data['instance_id']
            history = data.get('history') or []

            # Load completion file data (messages + tools)
            completion_data = load_last_completion(output_dir, instance_id)
            if completion_data is None:
                stats['no_completion'] += 1
                data['raw_completions'] = None
                f_out.write(json.dumps(data) + '\n')
                continue

            # Get the actual last response from output.jsonl history
            last_response = get_last_model_response(history)
            if last_response is None:
                stats['no_response'] += 1
                # Fall back to completion file's response
                data['raw_completions'] = None
                f_out.write(json.dumps(data) + '\n')
                continue

            # Check if it's oracle or solver
            # We detect oracle by checking tool_call IDs
            is_oracle = False
            for tc in (last_response.get('tool_calls') or []):
                if str(tc.get('id', '')).startswith('oracle_'):
                    is_oracle = True
                    break
            if is_oracle:
                stats['oracle_response'] += 1
            else:
                stats['solver_response'] += 1

            # Build the corrected raw_completions
            raw_completions = build_raw_completions(
                completion_data, last_response,
            )
            data['raw_completions'] = raw_completions
            f_out.write(json.dumps(data) + '\n')

    print(f'\nStats:')
    print(f'  Total instances: {stats["total"]}')
    print(f'  Oracle last response: {stats["oracle_response"]}')
    print(f'  Solver last response: {stats["solver_response"]}')
    print(f'  No completion files: {stats["no_completion"]}')
    print(f'  No model_response: {stats["no_response"]}')
    print(f'\nSaved to {output_path}')


if __name__ == '__main__':
    main()
