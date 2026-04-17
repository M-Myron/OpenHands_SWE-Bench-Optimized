"""
SWE-smith dataset utilities for OpenHands evaluation.

Handles loading, filtering, and adapting SWE-smith instances
so they can be used by the existing OpenHands evaluation pipeline.
"""

import json
import os
import re

import pandas as pd
from datasets import load_dataset

from openhands.core.logger import openhands_logger as logger


# SWE-smith HuggingFace dataset
SWESMITH_DATASET = 'SWE-bench/SWE-smith'
SWESMITH_SPLIT = 'train'

# Default test command for Python repos (pytest)
DEFAULT_PYTHON_TEST_CMD = (
    'source /opt/miniconda3/bin/activate && '
    'conda activate testbed && '
    'pytest --color=no -rA --tb=short'
)

# Test output markers used by SWE-smith eval
TEST_OUTPUT_START = '>>>>> Start Test Output'
TEST_OUTPUT_END = '>>>>> End Test Output'


def load_swesmith_dataset(
    dataset_name: str = SWESMITH_DATASET,
    split: str = SWESMITH_SPLIT,
    filter_real_only: bool = True,
    real_instances_path: str | None = None,
) -> pd.DataFrame:
    """Load SWE-smith dataset and optionally filter to real PR instances only.

    Args:
        dataset_name: HuggingFace dataset name
        split: dataset split
        filter_real_only: if True, filter to only PR-mirror instances
        real_instances_path: path to JSON file with real instance IDs.
            If None, uses the default file in split/ directory.

    Returns:
        DataFrame with SWE-smith instances
    """
    ds = load_dataset(dataset_name, split=split)
    df = ds.to_pandas()
    logger.info(f'Loaded SWE-smith dataset: {len(df)} total instances')

    if filter_real_only:
        if real_instances_path is None:
            real_instances_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'split',
                'swesmith_real_instances.json',
            )
        with open(real_instances_path, 'r') as f:
            real_ids = set(json.load(f))
        df = df[df['instance_id'].isin(real_ids)]
        logger.info(f'Filtered to {len(df)} real PR-mirror instances')

    return df


def get_swesmith_docker_image(instance: dict | pd.Series) -> str:
    """Get the Docker image name for a SWE-smith instance.

    SWE-smith instances have the image_name field directly in the dataset.
    """
    return instance['image_name']


def get_swesmith_workspace_dir_name(instance: dict | pd.Series) -> str:
    """Get the workspace directory name for an instance.

    SWE-smith repos are at /testbed in the container. This returns
    the directory name that will be symlinked to /workspace/.
    """
    return 'testbed'


def build_eval_script(instance: dict | pd.Series) -> str:
    """Build an evaluation script for a SWE-smith instance.

    Since SWE-smith doesn't use swebench's make_test_spec, we build
    the eval script ourselves from the FAIL_TO_PASS and PASS_TO_PASS lists.

    Args:
        instance: SWE-smith instance dict

    Returns:
        Shell script content as a string
    """
    f2p = instance.get('FAIL_TO_PASS', [])
    p2p = instance.get('PASS_TO_PASS', [])

    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)

    f2p = list(f2p)
    p2p = list(p2p)

    # Collect all test files from test paths
    test_files = set()
    for test_case in f2p + p2p:
        # Test cases are like "tests/path/test_file.py::TestClass::test_method"
        if '::' in test_case:
            test_files.add(test_case.split('::')[0])
        else:
            test_files.add(test_case)

    test_files_str = ' '.join(sorted(test_files))

    script = f"""#!/bin/bash
set -uxo pipefail
cd /testbed
source /opt/miniconda3/bin/activate
conda activate testbed
: '{TEST_OUTPUT_START}'
pytest --color=no -rA --tb=short -v {test_files_str} 2>&1
: '{TEST_OUTPUT_END}'
"""
    return script


def parse_pytest_log(log: str) -> dict[str, str]:
    """Parse pytest output to extract test status map.

    Args:
        log: Raw pytest output text

    Returns:
        Dict mapping test case names to status strings (PASSED/FAILED/ERROR/XFAIL)
    """
    test_status_map = {}
    for line in log.split('\n'):
        # Match lines like: tests/file.py::TestClass::test_method PASSED
        # or: PASSED tests/file.py::TestClass::test_method
        for status in ['PASSED', 'FAILED', 'ERROR', 'XFAIL', 'XPASS', 'SKIPPED']:
            match = re.match(rf'^(\S+)\s+{status}', line)
            if match:
                test_status_map[match.group(1)] = status
                break
    return test_status_map


def grade_swesmith_instance(
    instance: dict | pd.Series,
    test_output: str,
) -> dict:
    """Grade a SWE-smith prediction based on test output.

    The grading logic follows the same pattern as swebench:
    - All FAIL_TO_PASS tests must now PASS
    - No PASS_TO_PASS tests should FAIL

    Args:
        instance: SWE-smith instance with FAIL_TO_PASS, PASS_TO_PASS
        test_output: Raw test output from running eval script

    Returns:
        Report dict with resolved status and test details
    """
    report = {
        'resolved': False,
        'tests_status': {
            'FAIL_TO_PASS': {'success': [], 'failure': []},
            'PASS_TO_PASS': {'success': [], 'failure': []},
        },
    }

    # Extract test output between markers
    content = test_output
    start_marker = f"+ : '{TEST_OUTPUT_START}'"
    end_marker = f"+ : '{TEST_OUTPUT_END}'"
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)

    if start_idx >= 0 and end_idx > start_idx:
        content = content[start_idx + len(start_marker):end_idx]
    elif start_idx >= 0:
        content = content[start_idx + len(start_marker):]

    # Parse test statuses
    test_status_map = parse_pytest_log(content)

    f2p = instance.get('FAIL_TO_PASS', [])
    p2p = instance.get('PASS_TO_PASS', [])
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)
    f2p = list(f2p)
    p2p = list(p2p)

    # Check FAIL_TO_PASS: these should now PASS
    for test_case in f2p:
        if test_case in test_status_map and test_status_map[test_case] in ('PASSED', 'XFAIL'):
            report['tests_status']['FAIL_TO_PASS']['success'].append(test_case)
        elif test_case not in test_status_map or test_status_map[test_case] in ('FAILED', 'ERROR'):
            report['tests_status']['FAIL_TO_PASS']['failure'].append(test_case)
        # SKIPPED tests are not counted (matches SWE-smith behavior)

    # Check PASS_TO_PASS: these should still PASS
    for test_case in p2p:
        if test_case in test_status_map and test_status_map[test_case] in ('PASSED', 'XFAIL'):
            report['tests_status']['PASS_TO_PASS']['success'].append(test_case)
        elif test_case not in test_status_map or test_status_map[test_case] in ('FAILED', 'ERROR'):
            report['tests_status']['PASS_TO_PASS']['failure'].append(test_case)
        # SKIPPED tests are not counted (matches SWE-smith behavior)

    # Resolved = f2p_ratio == 1.0 AND p2p_ratio == 1.0
    # Following swebench's get_resolution_status: ratio = success/(success+failure), returns 1 if total=0
    f2p_total = len(report['tests_status']['FAIL_TO_PASS']['success']) + len(report['tests_status']['FAIL_TO_PASS']['failure'])
    p2p_total = len(report['tests_status']['PASS_TO_PASS']['success']) + len(report['tests_status']['PASS_TO_PASS']['failure'])
    f2p_ratio = len(report['tests_status']['FAIL_TO_PASS']['success']) / f2p_total if f2p_total > 0 else 1.0
    p2p_ratio = len(report['tests_status']['PASS_TO_PASS']['success']) / p2p_total if p2p_total > 0 else 1.0
    report['resolved'] = (f2p_ratio == 1.0 and p2p_ratio == 1.0)

    return report
