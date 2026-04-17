"""
SWE-smith evaluation script for OpenHands.

Evaluates agent predictions on SWE-smith instances by:
1. Loading predictions from output.jsonl
2. For each prediction, applying the git patch to the SWE-smith Docker container
3. Running the test suite
4. Grading based on FAIL_TO_PASS / PASS_TO_PASS criteria

Usage:
    poetry run python evaluation/benchmarks/swe_bench_optimized/swe_smith/eval_infer.py \
        --input-file <path/to/output.jsonl> \
        --dataset SWE-bench/SWE-smith \
        --split train
"""

import copy
import json
import os
import sys
import tempfile
import time

import pandas as pd
from tqdm import tqdm

from evaluation.benchmarks.swe_bench_optimized.swe_smith.dataset_utils import (
    build_eval_script,
    grade_swesmith_instance,
    load_swesmith_dataset,
)
from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    get_default_sandbox_config_for_eval,
    get_openhands_config_for_eval,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
)
from openhands.core.config import (
    LLMConfig,
    OpenHandsConfig,
    get_evaluation_parser,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime
from openhands.events.action import CmdRunAction
from openhands.events.observation import CmdOutputObservation
from openhands.utils.async_utils import call_async_from_sync


def process_git_patch(patch):
    """Clean up git patch output."""
    if not isinstance(patch, str):
        return ''
    if not patch.strip():
        return ''
    patch = patch.replace('\r\n', '\n')
    lines = patch.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('diff --git'):
            patch = '\n'.join(lines[i:])
            break
    patch = patch.rstrip() + '\n'
    return patch


def get_config(metadata: EvalMetadata, instance: pd.Series) -> OpenHandsConfig:
    """Get config using the SWE-smith image name from the instance."""
    base_container_image = instance['image_name']
    logger.info(f'Using SWE-smith container image: {base_container_image}')

    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = base_container_image
    sandbox_config.platform = 'linux/amd64'

    config = get_openhands_config_for_eval(
        runtime=os.environ.get('RUNTIME', 'docker'),
        sandbox_config=sandbox_config,
    )
    return config


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
    log_dir: str | None = None,
    runtime_failure_count: int = 0,
) -> EvalOutput:
    """Evaluate a single SWE-smith prediction."""
    if reset_logger:
        assert log_dir is not None
        os.makedirs(log_dir, exist_ok=True)
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)

    config = get_config(metadata, instance)
    instance_id = instance.instance_id
    model_patch = instance['model_patch']
    logger.info(f'Starting evaluation for instance {instance_id}')

    if 'test_result' not in instance.keys():
        instance['test_result'] = {}
    instance['test_result']['report'] = {
        'empty_generation': False,
        'resolved': False,
        'failed_apply_patch': False,
        'error_eval': False,
        'test_timeout': False,
    }

    if model_patch == '':
        instance['test_result']['report']['empty_generation'] = True
        return EvalOutput(
            instance_id=instance_id,
            test_result=instance['test_result'],
            metadata=metadata,
        )

    # Increase resource_factor for retries
    if runtime_failure_count > 0:
        config.sandbox.remote_runtime_resource_factor = min(
            config.sandbox.remote_runtime_resource_factor * (2**runtime_failure_count),
            8,
        )

    metadata = copy.deepcopy(metadata)
    metadata.details['runtime_failure_count'] = runtime_failure_count

    try:
        runtime = create_runtime(config)
        call_async_from_sync(runtime.connect)

        # SWE-smith: checkout the instance branch, then HEAD~1 to restore tests
        instance_branch = instance_id
        checkout_cmd = (
            f'cd /testbed && '
            f'git fetch origin {instance_branch} 2>/dev/null || true && '
            f'git checkout {instance_branch} 2>/dev/null && '
            f'git checkout HEAD~1 2>/dev/null && '
            f'echo "CHECKOUT_OK" || echo "CHECKOUT_SKIPPED"'
        )
        action = CmdRunAction(command=checkout_cmd)
        action.set_hard_timeout(600)
        obs = runtime.run_action(action)
        logger.info(f'Checkout result: {obs.content[:200] if hasattr(obs, "content") else obs}')

        # Copy patch to container
        with tempfile.TemporaryDirectory() as temp_dir:
            patch_file_path = os.path.join(temp_dir, 'patch.diff')
            with open(patch_file_path, 'w') as f:
                f.write(model_patch)
            runtime.copy_to(patch_file_path, '/tmp')

            # Build and copy eval script
            eval_script_content = build_eval_script(instance)
            eval_script_path = os.path.join(temp_dir, 'eval.sh')
            with open(eval_script_path, 'w') as f:
                f.write(eval_script_content)
            runtime.copy_to(eval_script_path, '/tmp')

        # Set +x on eval script
        action = CmdRunAction(command='chmod +x /tmp/eval.sh')
        action.set_hard_timeout(600)
        obs = runtime.run_action(action)
        assert obs.exit_code == 0

        # Apply patch
        exec_command = (
            'cd /testbed && '
            "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
            "(echo 'Failed to apply patch with git apply, trying with patch command...' && "
            "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
            "echo 'APPLY_PATCH_FAIL')))"
        )
        action = CmdRunAction(command=exec_command)
        action.set_hard_timeout(600)
        obs = runtime.run_action(action)
        assert isinstance(obs, CmdOutputObservation)
        apply_patch_output = obs.content
        instance['test_result']['apply_patch_output'] = apply_patch_output

        # CRITICAL: Revert any changes the prediction patch made to test files.
        # This prevents the agent from "cheating" by modifying tests to pass.
        # This matches SWE-smith's own eval harness behavior.
        f2p_tests = instance.get('FAIL_TO_PASS', [])
        p2p_tests = instance.get('PASS_TO_PASS', [])
        if isinstance(f2p_tests, str):
            import json as _json
            f2p_tests = _json.loads(f2p_tests)
        if isinstance(p2p_tests, str):
            import json as _json
            p2p_tests = _json.loads(p2p_tests)
        test_files_set = set()
        for tc in list(f2p_tests) + list(p2p_tests):
            if '::' in tc:
                test_files_set.add(tc.split('::')[0])
            else:
                test_files_set.add(tc)
        if test_files_set:
            revert_cmd = 'cd /testbed && git checkout -- ' + ' '.join(sorted(test_files_set))
            revert_action = CmdRunAction(command=revert_cmd)
            revert_action.set_hard_timeout(600)
            revert_obs = runtime.run_action(revert_action)
            logger.info(f'[{instance_id}] Reverted test files after patch apply')

        if 'APPLY_PATCH_FAIL' in apply_patch_output:
            logger.info(f'[{instance_id}] Patch apply failed:\n{apply_patch_output}')
            instance['test_result']['report']['failed_apply_patch'] = True
            return EvalOutput(
                instance_id=instance_id,
                test_result=instance['test_result'],
                metadata=metadata,
            )
        elif 'APPLY_PATCH_PASS' in apply_patch_output:
            logger.info(f'[{instance_id}] Patch applied successfully')

            # Run eval script in background
            log_file = '/tmp/eval_output.log'
            action = CmdRunAction(command=f'/tmp/eval.sh > {log_file} 2>&1 & echo $!')
            action.set_hard_timeout(300)
            obs = runtime.run_action(action)

            if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
                pid = obs.content.split()[-1].strip()
                logger.info(f'[{instance_id}] Eval started with PID: {pid}')

                # Poll for completion
                start_time = time.time()
                timeout = 1800  # 30 minutes
                while True:
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        logger.info(f'[{instance_id}] Eval timed out after {timeout}s')
                        instance['test_result']['report']['test_timeout'] = True
                        break
                    check_action = CmdRunAction(command=f'ps -p {pid} > /dev/null; echo $?')
                    check_action.set_hard_timeout(300)
                    check_obs = runtime.run_action(check_action)
                    if (
                        isinstance(check_obs, CmdOutputObservation)
                        and check_obs.content.split()[-1].strip() == '1'
                    ):
                        logger.info(f'[{instance_id}] Eval completed after {elapsed:.0f}s')
                        break
                    logger.info(f'[{instance_id}] [{elapsed:.0f}s] Eval still running...')
                    time.sleep(30)

                # Read test output
                cat_action = CmdRunAction(command=f'cat {log_file}')
                cat_action.set_hard_timeout(300)
                cat_obs = runtime.run_action(cat_action)

                if isinstance(cat_obs, CmdOutputObservation) and cat_obs.exit_code == 0:
                    test_output = cat_obs.content
                    instance['test_result']['test_output'] = test_output

                    # Grade the result using our SWE-smith grader
                    try:
                        report = grade_swesmith_instance(instance, test_output)
                        logger.info(
                            f'[{instance_id}] Result: resolved={report["resolved"]}'
                        )
                        instance['test_result']['report']['resolved'] = report['resolved']
                        instance['test_result']['report']['tests_status'] = report.get('tests_status', {})
                    except Exception as e:
                        logger.error(f'[{instance_id}] Grading error: {e}')
                        instance['test_result']['report']['error_eval'] = True
            else:
                logger.info(f'[{instance_id}] Error starting eval:\n{obs.content}')
                instance['test_result']['report']['error_eval'] = True

            return EvalOutput(
                instance_id=instance_id,
                test_result=instance['test_result'],
                metadata=metadata,
            )
        else:
            logger.info(f'[{instance_id}] Unexpected patch output:\n{apply_patch_output}')
            raise RuntimeError(f'Unexpected patch output for {instance_id}')
    finally:
        runtime.close()


if __name__ == '__main__':
    from functools import partial
    import subprocess

    parser = get_evaluation_parser()
    parser.add_argument(
        '--input-file',
        type=str,
        help='Path to input predictions file (output.jsonl)',
        required=True,
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='SWE-bench/SWE-smith',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
    )
    args, _ = parser.parse_known_args()

    # Load SWE-smith dataset for instance metadata
    swesmith_dataset = load_swesmith_dataset(
        dataset_name=args.dataset,
        split=args.split,
        filter_real_only=True,
    )
    instance_id_to_instance = {
        row['instance_id']: row.to_dict()
        for _, row in swesmith_dataset.iterrows()
    }
    logger.info(f'Loaded {len(instance_id_to_instance)} SWE-smith instances')

    # Output file
    output_file = args.input_file.replace('.jsonl', '.swebench_eval.jsonl')

    # Load already evaluated instances
    already_done = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'instance_id' in data:
                        already_done.add(str(data['instance_id']))
                except json.JSONDecodeError:
                    continue
        logger.info(f'Found {len(already_done)} already evaluated instances')

    # Load predictions
    assert args.input_file.endswith('.jsonl'), 'Input file must be a jsonl file.'
    records = []
    with open(args.input_file) as f:
        for line in tqdm(f, desc='Loading predictions'):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if str(obj.get('instance_id')) in already_done:
                    continue
                records.append({
                    'instance_id': obj['instance_id'],
                    'model_patch': obj.get('model_patch', obj.get('test_result', {}).get('git_patch', '')),
                    'test_result': obj.get('test_result', {}),
                })
            except (json.JSONDecodeError, KeyError):
                continue

    if not records:
        logger.info('No new predictions to evaluate. Exiting.')
        sys.exit(0)

    predictions = pd.DataFrame.from_records(records)
    predictions = predictions.drop_duplicates(subset=['instance_id'], keep='last')

    # Process model_patch
    predictions['model_patch'] = predictions['model_patch'].apply(process_git_patch)

    # Merge with dataset to get image_name, FAIL_TO_PASS, PASS_TO_PASS etc.
    predictions['image_name'] = predictions['instance_id'].apply(
        lambda x: instance_id_to_instance[x]['image_name']
    )
    predictions['FAIL_TO_PASS'] = predictions['instance_id'].apply(
        lambda x: instance_id_to_instance[x]['FAIL_TO_PASS']
    )
    predictions['PASS_TO_PASS'] = predictions['instance_id'].apply(
        lambda x: instance_id_to_instance[x]['PASS_TO_PASS']
    )
    predictions['repo'] = predictions['instance_id'].apply(
        lambda x: instance_id_to_instance[x]['repo']
    )

    # Prepare dataset
    instances = prepare_dataset(predictions, output_file, args.eval_n_limit)

    # Load or create metadata
    metadata: EvalMetadata | None = None
    metadata_filepath = os.path.join(os.path.dirname(args.input_file), 'metadata.json')
    if os.path.exists(metadata_filepath):
        with open(metadata_filepath, 'r') as metadata_file:
            metadata = EvalMetadata.model_validate_json(metadata_file.read())
    else:
        metadata = EvalMetadata(
            agent_class='dummy_agent',
            llm_config=LLMConfig(model='dummy_model'),
            max_iterations=1,
            eval_output_dir=os.path.dirname(args.input_file),
            start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'])
            .decode('utf-8').strip(),
            dataset=args.dataset,
            details={},
        )

    process_instance_func = partial(
        process_instance,
        log_dir=output_file.replace('.jsonl', '.logs'),
    )

    run_evaluation(
        instances,
        metadata=metadata,
        output_file=output_file,
        num_workers=args.eval_num_workers,
        process_instance_func=process_instance_func,
    )

    # Print summary
    evaluated = pd.read_json(output_file, lines=True)
    fields = ['resolved', 'failed_apply_patch', 'error_eval', 'empty_generation']
    for field in fields:
        count = evaluated.apply(
            lambda row: row['test_result']['report'][field], axis=1
        ).sum()
        logger.info(
            f'# {field}: {count} / {len(evaluated)} ({count / len(evaluated):.2%})'
        )
