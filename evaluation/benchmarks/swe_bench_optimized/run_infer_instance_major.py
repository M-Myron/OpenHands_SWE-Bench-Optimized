import asyncio
import copy
import errno
import hashlib
import json
import logging
import os
import queue
import resource
import subprocess
import time
import traceback
from multiprocessing import Manager, Pool
from typing import Any

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

import openhands.agenthub
from evaluation.benchmarks.swe_bench_optimized import run_infer as base
from evaluation.utils.shared import (
    EvalException,
    EvalMetadata,
    EvalOutput,
    EvalTimeoutException,
    check_maximum_retries_exceeded,
    is_fatal_runtime_error,
    log_skipped_maximum_retries_exceeded,
    make_metadata,
    prepare_dataset,
    timeout,
)
from openhands.core.config import (
    get_agent_config_arg,
    get_evaluation_parser,
    get_llm_config_arg,
)
from openhands.core.config.condenser_config import NoOpCondenserConfig
from openhands.core.config.utils import get_condenser_config_arg
from openhands.core.logger import get_console_handler
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.serialization.event import event_to_dict
from openhands.runtime.utils.runtime_build import (
    get_hash_for_lock_files,
    get_hash_for_source_files,
    get_runtime_image_repo,
    get_tag_for_versioned_image,
)
from openhands.utils.async_utils import call_async_from_sync
from openhands.version import get_version


# Worker globals initialized via _init_worker
_WORKER_N_RUNS = 1
_WORKER_SKIP_RUNS: set[int] = set()
_WORKER_RUN_CTX: dict[int, dict[str, Any]] = {}
_WORKER_RUNTIME = 'docker'
_WORKER_TIMEOUT_SECONDS: int | None = None
_WORKER_MAX_RETRIES = 5
_WORKER_DATASET_TYPE = 'SWE-bench'
_PROGRESS_QUEUE: Any = None

# Locks are process-safe objects passed through initializer
_OUTPUT_LOCKS: dict[int, Any] = {}


_RETRYABLE_CONTROLLER_ERROR_MARKERS = (
    'STATUS$ERROR_LLM_INTERNAL_SERVER_ERROR',
    'STATUS$ERROR_LLM_SERVICE_UNAVAILABLE',
)
_MAX_ITERATION_ERROR_MARKER = 'Agent reached maximum iteration'


def _is_retryable_controller_error(error: str | None) -> bool:
    """Whether controller-reported error should trigger retry semantics."""
    if not error:
        return False
    return any(marker in error for marker in _RETRYABLE_CONTROLLER_ERROR_MARKERS)


def _is_immediate_max_iteration_error(
    error: str | None,
    controller_elapsed_seconds: float,
) -> bool:
    """Detect stale-session max-iteration failures that happen right after startup."""
    if not error or _MAX_ITERATION_ERROR_MARKER not in error:
        return False
    threshold_seconds = float(
        os.environ.get('OH_EVAL_IMMEDIATE_MAX_ITERATION_THRESHOLD_SECONDS', '20')
    )
    return controller_elapsed_seconds <= threshold_seconds


def _make_eval_sid(instance_id: str, run_id: int, attempt_id: int = 0) -> str:
    """Build a deterministic and compact sid for container tracking.

    Docker container name becomes: openhands-runtime-<sid>.
    """
    execution_salt = os.environ.get('OH_EVAL_EXECUTION_SALT', '')
    digest = hashlib.sha1(
        f'{execution_salt}-{instance_id}-run-{run_id}-attempt-{attempt_id}'.encode(
            'utf-8'
        )
    ).hexdigest()[:18]
    sid = f'swg-r{run_id}a{attempt_id}-{digest}'
    return sid[:32]


def _runtime_images_for_base(base_image: str, enable_browser: bool) -> list[str]:
    """Compute candidate runtime image tags derived from a base image."""
    repo = get_runtime_image_repo()
    version = get_version()
    lock_tag = f'oh_v{version}_{get_hash_for_lock_files(base_image, enable_browser)}'
    source_tag = f'{lock_tag}_{get_hash_for_source_files()}'
    versioned_tag = f'oh_v{version}_{get_tag_for_versioned_image(base_image)}'
    # Runtime builder also supports a simple prebuilt tag based on base image.
    # Example: <repo>:docker.io_xingyaoww_sweb.eval.x86_64.<instance>
    simple_tag = base_image.replace('/', '_').replace(':', '_')
    return [
        f'{repo}:{source_tag}',
        f'{repo}:{lock_tag}',
        f'{repo}:{versioned_tag}',
        f'{repo}:{simple_tag}',
    ]


def _run_cleanup_cmd(cmd: list[str], desc: str):
    """Run cleanup command and surface failures for easier diagnosis."""
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or '').strip()
        logger.warning(
            f'Cleanup command failed ({desc}) with exit code {result.returncode}: '
            f'{stderr if stderr else "<no stderr>"}'
        )


def _cleanup_instance_docker_artifacts(
    instance_id: str,
    executed_sids: list[str],
    base_image: str,
    enable_browser: bool,
):
    """Best-effort cleanup after all runs of an instance are done."""
    logger.info(
        f'Cleaning docker artifacts for instance {instance_id}. '
        f'Containers tracked: {len(executed_sids)}'
    )

    # 1) Remove tracked runtime containers for this instance/runs.
    for sid in sorted(set(executed_sids)):
        container_name = f'openhands-runtime-{sid}'
        _run_cleanup_cmd(
            ['docker', 'rm', '-f', container_name],
            f'remove tracked container {container_name}',
        )

    # 2) Remove any containers using this base instance image.
    _run_cleanup_cmd(
        [
            'bash',
            '-lc',
            f'docker ps -aq --filter "ancestor={base_image}" | xargs -r docker rm -f',
        ],
        f'remove containers by base ancestor {base_image}',
    )

    # 3) Remove runtime images associated with this base image.
    for image in _runtime_images_for_base(base_image, enable_browser=enable_browser):
        _run_cleanup_cmd(
            ['docker', 'rmi', '-f', image],
            f'remove runtime image {image}',
        )

    # 3b) Remove any runtime images whose tag includes this base-image signature.
    # This catches additional prebuilt tag variants not covered above.
    simple_tag = base_image.replace('/', '_').replace(':', '_')
    _run_cleanup_cmd(
        [
            'bash',
            '-lc',
            (
                'docker images --format "{{.Repository}}:{{.Tag}}" '
                f'| grep -F "{simple_tag}" '
                '| xargs -r docker rmi -f'
            ),
        ],
        f'remove runtime images matching base signature {simple_tag}',
    )

    # 4) Remove instance base image.
    _run_cleanup_cmd(
        ['docker', 'rmi', '-f', base_image],
        f'remove base image {base_image}',
    )

    # 5) Free dangling image/build cache.
    _run_cleanup_cmd(
        ['docker', 'image', 'prune', '-f'],
        'prune dangling images',
    )
    _run_cleanup_cmd(
        ['docker', 'builder', 'prune', '-f', '--filter', 'until=1h'],
        'prune old builder cache',
    )


def _process_instance_with_sid(
    instance: pd.Series,
    metadata: EvalMetadata,
    sid: str,
    reset_logger: bool = True,
    runtime_failure_count: int = 0,
) -> EvalOutput:
    """Variant of base.process_instance that uses deterministic sid."""
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        _reset_worker_logger_quietly(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')

    config = base.get_config(instance, metadata)

    if runtime_failure_count > 0:
        config.sandbox.remote_runtime_resource_factor = min(
            config.sandbox.remote_runtime_resource_factor * (2**runtime_failure_count),
            8,
        )
        logger.warning(
            f'This is the {runtime_failure_count + 1}th attempt for instance {instance.instance_id}, '
            f'setting resource factor to {config.sandbox.remote_runtime_resource_factor}'
        )

    metadata = copy.deepcopy(metadata)
    metadata.details['runtime_failure_count'] = runtime_failure_count
    metadata.details['remote_runtime_resource_factor'] = (
        config.sandbox.remote_runtime_resource_factor
    )

    runtime = create_runtime(config, sid=sid)
    with base._env_prepare_concurrency_slot():
        call_async_from_sync(runtime.connect)
        base.initialize_runtime(runtime, instance, metadata)

    try:
        message_action = base.get_instruction(instance, metadata)

        controller_start = time.time()
        state = asyncio.run(
            run_controller(
                config=config,
                initial_user_action=message_action,
                runtime=runtime,
                fake_user_response_fn=base.AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[
                    metadata.agent_class
                ],
            )
        )
        controller_elapsed = time.time() - controller_start

        if base.is_fatal_evaluation_error(state.last_error):
            raise EvalException('Fatal error detected: ' + state.last_error)

        # Mirror retry/skip semantics for transient LLM transport failures:
        # convert controller-level status errors into retryable exceptions.
        if _is_retryable_controller_error(state.last_error):
            raise EvalException('Retryable controller error detected: ' + state.last_error)

        # If max-iteration is reached immediately after startup, this usually means
        # stale session state was resumed. Treat it as retryable/skip-eligible.
        if _is_immediate_max_iteration_error(state.last_error, controller_elapsed):
            raise EvalException(
                'Immediate max-iteration startup error detected '
                f'(elapsed={controller_elapsed:.2f}s): ' + state.last_error
            )

        if base.DATASET_TYPE == 'SWE-bench-Live':
            from evaluation.benchmarks.swe_bench.live_utils import (
                complete_runtime as complete_runtime_fn,
            )
        else:
            complete_runtime_fn = base.complete_runtime

        return_val = complete_runtime_fn(runtime, instance)
        git_patch = return_val['git_patch']
        logger.info(
            f'Got git diff for instance {instance.instance_id}:\n--------\n{git_patch}\n--------'
        )
    finally:
        runtime.close()

    test_result = {'git_patch': git_patch}

    if state is None:
        raise ValueError('State should not be None.')

    histories = [event_to_dict(event) for event in state.history]
    metrics = base.get_metrics(state)

    instruction = message_action.content
    if message_action.image_urls:
        instruction += (
            '\n\n<image_urls>' + '\n'.join(message_action.image_urls) + '</image_urls>'
        )

    return EvalOutput(
        instance_id=instance.instance_id,
        instruction=instruction,
        instance=instance.to_dict(),
        test_result=test_result,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
    )


def _run_single_with_retries(
    instance: pd.Series,
    metadata: EvalMetadata,
    run_id: int,
    use_mp: bool,
    max_retries: int,
    timeout_seconds: int | None,
    executed_sids: list[str] | None = None,
) -> EvalOutput:
    """Retry wrapper adapted for deterministic sid processing."""
    runtime_failure_count = 0
    instance_id = str(instance.instance_id)
    for attempt in range(max_retries + 1):
        sid = _make_eval_sid(instance_id, run_id, attempt)
        if executed_sids is not None and sid not in executed_sids:
            executed_sids.append(sid)
        try:
            if timeout_seconds is not None:
                with timeout(timeout_seconds):
                    return _process_instance_with_sid(
                        instance,
                        metadata,
                        sid=sid,
                        reset_logger=use_mp,
                        runtime_failure_count=runtime_failure_count,
                    )
            return _process_instance_with_sid(
                instance,
                metadata,
                sid=sid,
                reset_logger=use_mp,
                runtime_failure_count=runtime_failure_count,
            )
        except EvalTimeoutException as e:
            error = f'Timeout after {timeout_seconds} seconds'
            stacktrace = traceback.format_exc()
            msg = (
                '-' * 10
                + '\n'
                + f'Timeout ({timeout_seconds} seconds) in instance [{instance.instance_id}], Stopped evaluation for this instance.'
                + '\n'
                + '-' * 10
            )
            logger.exception(e)
            return EvalOutput(
                instance_id=instance.instance_id,
                test_result={},
                error=error,
            )
        except Exception as e:
            error = str(e)
            stacktrace = traceback.format_exc()

            if attempt == max_retries:
                msg = (
                    '-' * 10
                    + '\n'
                    + f'Error in instance [{instance.instance_id}]: {error}. Stacktrace:\n{stacktrace}'
                    + '\n'
                    + f'[Encountered after {max_retries} retries. Please check the logs and report the issue.]'
                    + '-' * 10
                )
                skip_errors = (
                    os.environ.get(
                        'EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED', 'false'
                    ).lower()
                    == 'true'
                )
                if skip_errors:
                    return log_skipped_maximum_retries_exceeded(
                        instance, metadata, e, max_retries
                    )

                logger.exception(e)
                raise RuntimeError(
                    f'Maximum error retries reached for instance {instance.instance_id}'
                ) from e

            msg = (
                '-' * 10
                + '\n'
                + f'Error in instance [{instance.instance_id}]: {error}. Stacktrace:\n{stacktrace}'
                + '\n'
                + '-' * 10
                + f'[The above error occurred. Retrying... (attempt {attempt + 1} of {max_retries})]'
                + '-' * 10
                + '\n'
            )
            _error_str = type(e).__name__ + ': ' + str(e)
            if is_fatal_runtime_error(_error_str):
                runtime_failure_count += 1
                msg += (
                    f'Runtime disconnected error detected for instance {instance.instance_id}, '
                    f'runtime failure count: {runtime_failure_count}'
                )
                msg += '\n' + '-' * 10 + '\n'
            logger.error(msg)
            time.sleep(5)

    raise RuntimeError('unreachable')


def _is_max_retry_skipped(result: EvalOutput) -> bool:
    if not result.error:
        return False
    return str(result.error).lower().startswith('maximum retries')


def _init_worker(
    n_runs: int,
    skip_runs: set[int],
    run_ctx: dict[int, dict[str, Any]],
    runtime: str,
    timeout_seconds: int | None,
    max_retries: int,
    dataset_type: str,
    output_locks: dict[int, Any],
    progress_queue: Any,
):
    global _WORKER_N_RUNS
    global _WORKER_SKIP_RUNS
    global _WORKER_RUN_CTX
    global _WORKER_RUNTIME
    global _WORKER_TIMEOUT_SECONDS
    global _WORKER_MAX_RETRIES
    global _WORKER_DATASET_TYPE
    global _OUTPUT_LOCKS
    global _PROGRESS_QUEUE

    _WORKER_N_RUNS = n_runs
    _WORKER_SKIP_RUNS = skip_runs
    _WORKER_RUN_CTX = run_ctx
    _WORKER_RUNTIME = runtime
    _WORKER_TIMEOUT_SECONDS = timeout_seconds
    _WORKER_MAX_RETRIES = max_retries
    _WORKER_DATASET_TYPE = dataset_type
    _OUTPUT_LOCKS = output_locks
    _PROGRESS_QUEUE = progress_queue

    # Ensure helper functions in base module compute image naming consistently.
    base.DATASET_TYPE = dataset_type


def _append_result(output_file: str, result: EvalOutput, run_id: int):
    lock = _OUTPUT_LOCKS[run_id]
    with lock:
        with open(output_file, 'a') as f:
            f.write(result.model_dump_json() + '\n')
            f.flush()


def _reset_worker_logger_quietly(
    target_logger: logging.Logger,
    instance_id: str,
    log_dir: str,
):
    """Like reset_logger_for_multiprocessing but without noisy startup console lines."""
    log_file = os.path.join(log_dir, f'instance_{instance_id}.log')

    # IMPORTANT: removeHandler() alone does not close file descriptors.
    # Explicitly close old handlers to avoid leaking FDs across many instances.
    for handler in target_logger.handlers[:]:
        target_logger.removeHandler(handler)
        try:
            handler.flush()
        except Exception:
            pass
        try:
            handler.close()
        except Exception:
            pass

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    worker_console_level = os.environ.get('OH_EVAL_WORKER_CONSOLE_LEVEL', 'ERROR')
    console_level = getattr(logging, worker_console_level.upper(), logging.ERROR)

    console_handler = get_console_handler(log_level=console_level)
    console_handler.setFormatter(
        logging.Formatter(
            f'Instance {instance_id} - ' + '%(asctime)s - %(levelname)s - %(message)s'
        )
    )
    console_handler.setLevel(console_level)
    target_logger.addHandler(console_handler)

    try:
        file_handler = logging.FileHandler(log_file)
    except OSError as e:
        # Under heavy load, we may transiently hit EMFILE. Keep worker alive
        # with console logging so retries/cleanup can still proceed.
        if e.errno == errno.EMFILE:
            target_logger.warning(
                'File logging disabled for instance %s due to open-file limit (EMFILE): %s',
                instance_id,
                e,
            )
            return
        raise

    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    file_handler.setLevel(logging.INFO)
    target_logger.addHandler(file_handler)


def _process_instance_all_runs(instance_dict: dict[str, Any]) -> dict[str, Any]:
    """Process one instance across all runs sequentially.

    This gives the desired order per worker:
    instance_i_run1 -> instance_i_run2 -> ... -> instance_i_runN.
    """
    instance = pd.Series(instance_dict)
    instance_id = str(instance.instance_id)

    executed_sids: list[str] = []
    processed_runs: list[int] = []
    skipped_runs: list[int] = []

    for run_id in range(1, _WORKER_N_RUNS + 1):
        if run_id in _WORKER_SKIP_RUNS:
            skipped_runs.append(run_id)
            continue

        ctx = _WORKER_RUN_CTX[run_id]
        completed_ids = set(ctx['completed_ids'])
        if instance_id in completed_ids:
            skipped_runs.append(run_id)
            continue

        metadata = EvalMetadata.model_validate_json(ctx['metadata_json'])
        output_file = ctx['output_file']
        run_output_dir = ctx['run_output_dir']
        infer_log_file = os.path.join(
            run_output_dir, 'infer_logs', f'instance_{instance_id}.log'
        )

        if _PROGRESS_QUEUE is not None:
            _PROGRESS_QUEUE.put(
                {
                    'type': 'run_started',
                    'instance_id': instance_id,
                    'run_id': run_id,
                    'infer_log_file': infer_log_file,
                }
            )

        result = _run_single_with_retries(
            instance=instance,
            metadata=metadata,
            run_id=run_id,
            use_mp=True,
            max_retries=_WORKER_MAX_RETRIES,
            timeout_seconds=_WORKER_TIMEOUT_SECONDS,
            executed_sids=executed_sids,
        )
        _append_result(output_file, result, run_id)
        processed_runs.append(run_id)
        run_status = 'skipped' if _is_max_retry_skipped(result) else (
            'failed' if bool(result.error) else 'finished'
        )
        if _PROGRESS_QUEUE is not None:
            _PROGRESS_QUEUE.put(
                {
                    'type': 'run_finished',
                    'instance_id': instance_id,
                    'run_id': run_id,
                    'error': bool(result.error),
                    'status': run_status,
                    'test_result_preview': str(result.test_result)[:300],
                }
            )
        if run_status == 'skipped':
            logger.info(
                f'### SKIPPED RUN {run_id} FOR INSTANCE {instance_id} '
                f'(maximum retries exceeded) ###'
            )
        else:
            logger.info(
                f'### FINISHED RUN {run_id} FOR INSTANCE {instance_id} '
                f'(error={bool(result.error)}) ###'
            )

    # Once this instance has completed all pending runs, clean its docker artifacts.
    if _WORKER_RUNTIME == 'docker' and (processed_runs or skipped_runs):
        use_swebench_official_image = _WORKER_DATASET_TYPE != 'SWE-Gym'
        base_image = base.get_instance_docker_image(
            instance_id,
            swebench_official_image=use_swebench_official_image,
        )
        _cleanup_instance_docker_artifacts(
            instance_id=instance_id,
            executed_sids=executed_sids,
            base_image=base_image,
            enable_browser=base.RUN_WITH_BROWSING,
        )

    return {
        'instance_id': instance_id,
        'processed_runs': processed_runs,
        'skipped_runs': skipped_runs,
    }


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f'{int(seconds)}s'
    if seconds < 3600:
        return f'{int(seconds // 60)}m {int(seconds % 60)}s'
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f'{hours}h {minutes}m'


def _ensure_open_file_limit(min_soft_limit: int = 65535) -> None:
    """Best-effort increase of RLIMIT_NOFILE to reduce EMFILE under high concurrency."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception as e:
        logger.warning('Failed to read RLIMIT_NOFILE: %s', e)
        return

    target_soft = soft
    if hard == resource.RLIM_INFINITY:
        target_soft = max(soft, min_soft_limit)
    else:
        target_soft = min(max(soft, min_soft_limit), hard)

    if target_soft > soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target_soft, hard))
            soft = target_soft
        except Exception as e:
            logger.warning(
                'Failed to raise RLIMIT_NOFILE soft limit from %s to %s (hard=%s): %s',
                soft,
                target_soft,
                hard,
                e,
            )

    logger.info('RLIMIT_NOFILE configured to soft=%s hard=%s', soft, hard)


if __name__ == '__main__':
    _ensure_open_file_limit()

    parser = get_evaluation_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='princeton-nlp/SWE-bench_Verified',
        help='data set to evaluate on, either full-test or lite-test',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='split to evaluate on',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='swe',
        choices=['swe', 'swt', 'swt-ci'],
        help="mode to run the evaluation, either 'swe', 'swt', or 'swt-ci'",
    )
    parser.add_argument(
        '--n-runs',
        type=int,
        default=1,
        help='number of runs per instance',
    )

    args, _ = parser.parse_known_args()

    # Ensure each top-level rollout run uses a unique SID namespace so workers
    # don't resume stale controller sessions from previous executions.
    if not os.environ.get('OH_EVAL_EXECUTION_SALT'):
        salt_seed = f'{time.time_ns()}-{os.getpid()}'
        os.environ['OH_EVAL_EXECUTION_SALT'] = hashlib.sha1(
            salt_seed.encode('utf-8')
        ).hexdigest()[:10]
    logger.info(
        'Using OH_EVAL_EXECUTION_SALT=%s for SID namespacing.',
        os.environ['OH_EVAL_EXECUTION_SALT'],
    )

    dataset = load_dataset(args.dataset, split=args.split)
    base.set_dataset_type(args.dataset)

    swe_bench_tests = base.filter_dataset(dataset.to_pandas(), 'instance_id')
    logger.info(
        f'Loaded dataset {args.dataset} with split {args.split}: {len(swe_bench_tests)} tasks'
    )

    if base.DATASET_TYPE == 'SWE-Gym':
        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'split',
                'swegym_verified_instances.json',
            ),
            'r',
        ) as f:
            swegym_verified_instances = json.load(f)
            swe_bench_tests = swe_bench_tests[
                swe_bench_tests['instance_id'].isin(swegym_verified_instances)
            ]
        logger.info(
            f'{len(swe_bench_tests)} tasks left after filtering for SWE-Gym verified instances'
        )

    if len(swe_bench_tests) > 0 and not isinstance(
        swe_bench_tests['PASS_TO_PASS'][swe_bench_tests['PASS_TO_PASS'].index[0]], str
    ):
        for col in ['PASS_TO_PASS', 'FAIL_TO_PASS']:
            swe_bench_tests[col] = swe_bench_tests[col].apply(lambda x: str(x))

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config, args.config_file)
        llm_config.log_completions = True
        llm_config.modify_params = False

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    condenser_name = os.environ.get('EVAL_CONDENSER')
    if condenser_name:
        condenser_config = get_condenser_config_arg(condenser_name, args.config_file)
        if condenser_config is None:
            raise ValueError(
                f'Could not find Condenser config: EVAL_CONDENSER={condenser_name}'
            )
    else:
        condenser_config = NoOpCondenserConfig()
        logger.debug(
            'No Condenser config provided via EVAL_CONDENSER, using NoOpCondenser.'
        )

    agent_config = None
    if args.agent_config:
        agent_config = get_agent_config_arg(args.agent_config, args.config_file)

    details = {'mode': args.mode}
    _agent_cls = openhands.agenthub.Agent.get_cls(args.agent_cls)

    dataset_description = (
        args.dataset.replace('/', '__') + '-' + args.split.replace('/', '__')
    )

    n_runs = args.n_runs
    skip_runs_str = os.environ.get('SKIP_RUNS', '')
    skip_runs = set(int(x.strip()) for x in skip_runs_str.split(',') if x.strip())
    active_runs = [r for r in range(1, n_runs + 1) if r not in skip_runs]

    if not active_runs:
        logger.warning('All runs are skipped via SKIP_RUNS; nothing to do.')
        raise SystemExit(0)

    run_context: dict[int, dict[str, Any]] = {}
    union_pending_ids: set[str] = set()

    for run_id in range(1, n_runs + 1):
        if run_id in skip_runs:
            continue

        run_eval_note = args.eval_note if n_runs == 1 else f'{args.eval_note}-run_{run_id}'
        run_metadata = make_metadata(
            llm_config,
            dataset_description,
            args.agent_cls,
            args.max_iterations,
            run_eval_note,
            args.eval_output_dir,
            details=details,
            agent_config=agent_config,
            condenser_config=condenser_config,
        )
        run_output_file = os.path.join(run_metadata.eval_output_dir, 'output.jsonl')

        if n_runs > 1:
            print(f'### OUTPUT FILE FOR RUN {run_id}: {run_output_file} ###')
        else:
            print(f'### OUTPUT FILE: {run_output_file} ###')

        run_instances = prepare_dataset(swe_bench_tests, run_output_file, args.eval_n_limit)
        pending_ids = set(str(x) for x in run_instances['instance_id'].tolist())

        run_context[run_id] = {
            'metadata_json': run_metadata.model_dump_json(),
            'output_file': run_output_file,
            'run_output_dir': run_metadata.eval_output_dir,
            'completed_ids': sorted(
                set(str(x) for x in swe_bench_tests['instance_id'].tolist()) - pending_ids
            ),
            'pending_count': len(pending_ids),
        }

        union_pending_ids.update(pending_ids)

    instance_tasks: list[dict[str, Any]] = []
    for _, row in swe_bench_tests.iterrows():
        if str(row['instance_id']) in union_pending_ids:
            instance_tasks.append(row.to_dict())

    total_instances_to_process = len(instance_tasks)
    logger.info('=' * 80)
    logger.info('EVALUATION PLAN (INSTANCE-MAJOR):')
    logger.info(f'  Total instances in dataset: {len(swe_bench_tests)}')
    logger.info(f'  Instances with pending runs: {total_instances_to_process}')
    logger.info(f'  Runs per instance: {n_runs}')
    logger.info(
        f'  Active runs: {len(active_runs)} (skipping: {skip_runs if skip_runs else "none"})'
    )
    logger.info(f'  Worker processes: {args.eval_num_workers}')
    for run_id in active_runs:
        run_pending = run_context[run_id]['pending_count']
        run_completed = len(run_context[run_id]['completed_ids'])
        logger.info(
            f'  Run {run_id}: completed={run_completed}, pending={run_pending}'
        )
    logger.info('=' * 80)

    if total_instances_to_process == 0:
        logger.info('No pending instances across all runs. Done.')
        raise SystemExit(0)

    manager = Manager()
    output_locks = {run_id: manager.Lock() for run_id in active_runs}
    progress_queue = manager.Queue()

    total_pending_evaluations = sum(
        int(run_context[run_id]['pending_count']) for run_id in active_runs
    )
    logger.info(
        f'  Total pending evaluations across all runs: {total_pending_evaluations}'
    )

    start = time.time()
    completed_instances = 0

    with Pool(
        processes=args.eval_num_workers,
        initializer=_init_worker,
        initargs=(
            n_runs,
            skip_runs,
            run_context,
            os.environ.get('RUNTIME', 'docker'),
            8 * 60 * 60,
            5,
            base.DATASET_TYPE,
            output_locks,
            progress_queue,
        ),
    ) as pool:
        async_results = [
            pool.apply_async(_process_instance_all_runs, (task,)) for task in instance_tasks
        ]

        with tqdm(total=total_pending_evaluations, desc='Evaluations processed') as pbar:
            pending_results = async_results
            while pending_results:
                drained_any = False

                while True:
                    try:
                        event = progress_queue.get_nowait()
                    except queue.Empty:
                        break

                    drained_any = True
                    if event.get('type') == 'run_started':
                        logger.info(
                            f"Starting evaluation for instance {event.get('instance_id')} "
                            f"(run {event.get('run_id')}/{n_runs})."
                        )
                        logger.info(
                            f"Hint: run \"tail -f {event.get('infer_log_file')}\" "
                            f'to see live logs in a separate shell'
                        )
                    elif event.get('type') == 'run_finished':
                        pbar.update(1)
                        pbar.set_description(
                            f"Instance {event.get('instance_id')}"
                        )
                        pbar.set_postfix_str(
                            f"Run {event.get('run_id')}/{n_runs} "
                            f"error={event.get('error')} "
                            f"Test Result: {event.get('test_result_preview')}"
                        )
                        logger.info(
                            f"{event.get('status', 'finished').upper()} evaluation for instance "
                            f"{event.get('instance_id')} (run {event.get('run_id')}): "
                            f"{event.get('test_result_preview')}"
                        )

                next_pending: list[Any] = []
                for ar in pending_results:
                    if ar.ready():
                        summary = ar.get()
                        completed_instances += 1

                        elapsed = time.time() - start
                        avg = elapsed / completed_instances
                        eta = avg * (total_instances_to_process - completed_instances)
                        logger.info(
                            f"Completed instance {summary['instance_id']} | "
                            f'instance progress {completed_instances}/{total_instances_to_process} | '
                            f'elapsed {_format_time(elapsed)} | eta {_format_time(eta)}'
                        )
                        logger.info(
                            f"### FINISHED INSTANCE {summary['instance_id']} | "
                            f"processed_runs={summary['processed_runs']} "
                            f"skipped_runs={summary['skipped_runs']} ###"
                        )
                    else:
                        next_pending.append(ar)

                pending_results = next_pending
                if pending_results and not drained_any:
                    time.sleep(0.5)

            # Drain any late queue events before closing progress output.
            while True:
                try:
                    event = progress_queue.get_nowait()
                except queue.Empty:
                    break
                if event.get('type') == 'run_finished':
                    pbar.update(1)
                    pbar.set_description(f"Instance {event.get('instance_id')}")
                    pbar.set_postfix_str(
                        f"Run {event.get('run_id')}/{n_runs} "
                        f"error={event.get('error')} "
                        f"Test Result: {event.get('test_result_preview')}"
                    )
                    logger.info(
                        f"{event.get('status', 'finished').upper()} evaluation for instance "
                        f"{event.get('instance_id')} (run {event.get('run_id')}): "
                        f"{event.get('test_result_preview')}"
                    )
                elif event.get('type') == 'run_started':
                    logger.info(
                        f"Starting evaluation for instance {event.get('instance_id')} "
                        f"(run {event.get('run_id')}/{n_runs})."
                    )
                    logger.info(
                        f"Hint: run \"tail -f {event.get('infer_log_file')}\" "
                        f'to see live logs in a separate shell'
                    )

    total_time = time.time() - start
    logger.info('')
    logger.info('╔═══════════════════════════════════════════════════════════════════════════╗')
    logger.info('║ ALL EVALUATIONS COMPLETE!')
    logger.info(
        f'║ Processed instances: {completed_instances}/{total_instances_to_process} '
        f'| Total time: {_format_time(total_time)}'
    )
    logger.info('╚═══════════════════════════════════════════════════════════════════════════╝')
    logger.info('')

    for run_id in active_runs:
        output_file = run_context[run_id]['output_file']
        eval_output_dir = os.path.dirname(output_file)
        check_maximum_retries_exceeded(eval_output_dir)
