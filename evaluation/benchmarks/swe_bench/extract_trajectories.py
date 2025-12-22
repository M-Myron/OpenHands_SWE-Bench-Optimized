#!/usr/bin/env python3
"""
Extract trajectories from SWE-bench inference output.

This script reads the output.jsonl file from SWE-bench inference runs
and extracts individual trajectory files for each instance.

Usage:
    python extract_trajectories.py --input-file output.jsonl --output-dir trajectories/

    # Or extract from evaluation output directory
    python extract_trajectories.py --eval-output-dir evaluation/evaluation_outputs/xxx --output-dir trajectories/
"""

import argparse
import json
import os
from pathlib import Path


def extract_trajectories(
    input_file: str,
    output_dir: str,
    overwrite: bool = False
):
    """
    Extract trajectories from output.jsonl and save as individual JSON files.

    Args:
        input_file: Path to output.jsonl file
        output_dir: Directory to save trajectory files
        overwrite: Whether to overwrite existing trajectory files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    extracted_count = 0
    skipped_count = 0
    error_count = 0

    print(f"Reading from: {input_file}")
    print(f"Saving to: {output_dir}")
    print("-" * 80)

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse the line
                data = json.loads(line.strip())
                instance_id = data.get('instance_id')

                if not instance_id:
                    print(f"Warning: Line {line_num} has no instance_id, skipping")
                    error_count += 1
                    continue

                # Create trajectory file path
                trajectory_file = os.path.join(output_dir, f"{instance_id}.json")

                # Check if file exists
                if os.path.exists(trajectory_file) and not overwrite:
                    print(f"Skipping {instance_id}: trajectory file already exists")
                    skipped_count += 1
                    continue

                # Save the entire output (includes history, metadata, test_result, etc.)
                with open(trajectory_file, 'w') as out_f:
                    json.dump(data, out_f, indent=2)

                extracted_count += 1
                print(f"✓ Extracted trajectory for {instance_id}")

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                error_count += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                error_count += 1

    print("-" * 80)
    print(f"Summary:")
    print(f"  - Extracted: {extracted_count} trajectories")
    print(f"  - Skipped: {skipped_count} (already exist)")
    print(f"  - Errors: {error_count}")
    print(f"\nTrajectory files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract trajectories from SWE-bench inference output"
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input-file',
        type=str,
        help='Path to output.jsonl file'
    )
    input_group.add_argument(
        '--eval-output-dir',
        type=str,
        help='Path to evaluation output directory (will look for output.jsonl inside)'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='trajectories',
        help='Directory to save trajectory files (default: trajectories/)'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing trajectory files'
    )

    args = parser.parse_args()

    # Determine input file path
    if args.eval_output_dir:
        input_file = os.path.join(args.eval_output_dir, 'output.jsonl')
        if not os.path.exists(input_file):
            print(f"Error: output.jsonl not found in {args.eval_output_dir}")
            return 1
    else:
        input_file = args.input_file
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}")
            return 1

    # Extract trajectories
    extract_trajectories(
        input_file=input_file,
        output_dir=args.output_dir,
        overwrite=args.overwrite
    )

    return 0


if __name__ == '__main__':
    exit(main())
