"""Instance-major Oracle-Guided V2 evaluation runner.

Schedules work as: instance1_run1..runN, instance2_run1..runN, ...
This avoids waiting for the slowest instance in a batch before the next
batch starts—every worker processes one instance through all runs and
immediately picks up the next available instance.

Drop-in replacement for ``run_infer_oracle_guided_v2.py`` with better
throughput when NUM_WORKERS < total instances.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import queue
import resource
import time
import traceback
from multiprocessing import Manager, Pool
from typing import Any

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

import openhands.agenthub  # noqa: F401
import openhands.agenthub.oracle_guided_v2_codeact_agent  # noqa: F401
from evaluation.benchmarks.swe_bench_optimized import run_infer as base
from evaluation.benchmarks.swe_bench_optimized.run_infer import (
    AGENT_CLS_TO_FAKE_USER_RESPONSE_FN,
)
from evaluation.benchmarks.swe_bench_optimized.run_infer_oracle_guided_v2 import (
    _build_issue_understanding,
    _load_react_facts,
    _write_oracle_context_file,
)
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
from openhands.agenthub.oracle_guided_v2_codeact_agent.oracle_guided_v2_codeact_agent import (
    clear_triage_log,
    read_and_clear_triage_log,
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
from openhands.utils.async_utils import call_async_from_sync

# Re-use the instance-major utilities from the base instance-major runner
from evaluation.benchmarks.swe_bench_optimized.run_infer_instance_major import (
    _cleanup_instance_docker_artifacts,
    _make_eval_sid,
    _reset_worker_logger_quietly,
)


# ---------------------------------------------------------------------------
# Fake user response registration
# ---------------------------------------------------------------------------
AGENT_CLS_TO_FAKE_USER_RESPONSE_FN['OracleGuidedV2CodeActAgent'] = (
    AGENT_CLS_TO_FAKE_USER_RESPONSE_FN.get('CodeActAgent')
)


# ---------------------------------------------------------------------------
# Worker globals (set via _init_worker)
# ---------------------------------------------------------------------------
_WORKER_N_RUNS = 1
_WORKER_SKIP_RUNS: set[int] = set()
_WORKER_RUN_CTX: dict[int, dict[str, Any]] = {}
_WORKER_RUNTIME = 'docker'
_WORKER_TIMEOUT_SECONDS: int | None = None
_WORKER_MAX_RETRIES = 5
_WORKER_DATASET_TYPE = 'SWE-bench'
_PROGRESS_QUEUE: Any = None
_OUTPUT_LOCKS: dict[int, Any] = {}


# ---------------------------------------------------------------------------
# Retry / error helpers (mirrored from instance_major)
# ---------------------------------------------------------------------------
_RETRYABLE_CONTROLLER_ERROR_MARKERS = (
    'STATUS$ERROR_LLM_INTERNAL_SERVER_ERROR',
    'STATUS$ERROR_LLM_SERVICE_UNAVAILABLE',
)
_MAX_ITERATION_ERROR_MARKER = 'Agent reached maximum iteration'


def _is_retryable_controller_error(error: str | None) -> bool:
    if not error:
        return False
    return any(m in error for m in _RETRYABLE_CONTROLLER_ERROR_MARKERS)


def _is_immediate_max_iteration_error(
    error: str | None, controller_elapsed_seconds: float
) -> bool:
    if not error or _MAX_ITERATION_ERROR_MARKER not in error:
        return False
    threshold = float(
        os.environ.get('OH_EVAL_IMMEDIATE_MAX_ITERATION_THRESHOLD_SECONDS', '20')
    )
    return controller_elapsed_seconds <= threshold


# ---------------------------------------------------------------------------
# Core per-instance processing (Oracle-Guided V2 specific)
# ---------------------------------------------------------------------------

def _process_oracle_guided_v2_instance_with_sid(
    instance: pd.Series,
    metadata: EvalMetadata,
    sid: str,
    reset_logger: bool = True,
    runtime_failure_count: int = 0,
) -> EvalOutput:
    """Process one instance with Oracle-Guided V2 context setup + deterministic sid."""
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        _reset_worker_logger_quietly(logger, instance.instance_id, log_dir)

    # ---- Oracle-Guided V2 context setup ----
    clear_triage_log()
    context_path = _write_oracle_context_file(instance, metadata)
    os.environ['ORACLE_GUIDED_CONTEXT_PATH'] = context_path

    save_planner = os.environ.get('GUIDED_V2_SAVE_PLANNER_PROMPTS', '1').strip() == '1'
    if save_planner:
        os.environ['GUIDED_V2_PLANNER_SAVE_PROMPTS_DIR'] = os.path.join(
            metadata.eval_output_dir, 'oracle_guided_v2_planner_prompts', str(instance.instance_id),
        )
    else:
        os.environ.pop('GUIDED_V2_PLANNER_SAVE_PROMPTS_DIR', None)

    save_critic = os.environ.get('GUIDED_V2_SAVE_CRITIC_PROMPTS', '1').strip() == '1'
    if save_critic:
        critic_dir = os.path.join(
            metadata.eval_output_dir, 'oracle_guided_v2_critic_prompts', str(instance.instance_id),
        )
        os.environ['GUIDED_V2_SUFFICIENCY_CRITIC_SAVE_PROMPTS_DIR'] = critic_dir
        os.environ['GUIDED_V2_LEAKAGE_CRITIC_SAVE_PROMPTS_DIR'] = critic_dir
    else:
        os.environ.pop('GUIDED_V2_SUFFICIENCY_CRITIC_SAVE_PROMPTS_DIR', None)
        os.environ.pop('GUIDED_V2_LEAKAGE_CRITIC_SAVE_PROMPTS_DIR', None)

    # ---- Standard SWE-bench instance processing ----
    config = base.get_config(instance, metadata)

    if runtime_failure_count > 0:
        config.sandbox.remote_runtime_resource_factor = min(
            config.sandbox.remote_runtime_resource_factor * (2 ** runtime_failure_count), 8,
        )

    metadata = copy.deepcopy(metadata)
    metadata.details['runtime_failure_count'] = runtime_failure_count

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
                fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[metadata.agent_class],
            )
        )
        controller_elapsed = time.time() - controller_start

        if base.is_fatal_evaluation_error(state.last_error):
            raise EvalException('Fatal error detected: ' + state.last_error)
        if _is_retryable_controller_error(state.last_error):
            raise EvalException('Retryable controller error: ' + state.last_error)
        if _is_immediate_max_iteration_error(state.last_error, controller_elapsed):
            raise EvalException('Immediate max-iteration error: ' + state.last_error)

        return_val = base.complete_runtime(runtime, instance)
        git_patch = return_val['git_patch']
    finally:
        runtime.close()
        os.environ.pop('ORACLE_GUIDED_CONTEXT_PATH', None)
        os.environ.pop('GUIDED_V2_PLANNER_SAVE_PROMPTS_DIR', None)
        os.environ.pop('GUIDED_V2_SUFFICIENCY_CRITIC_SAVE_PROMPTS_DIR', None)
        os.environ.pop('GUIDED_V2_LEAKAGE_CRITIC_SAVE_PROMPTS_DIR', None)

    # ---- Collect guided log ----
    guided_log = read_and_clear_triage_log()
    guided_log_dir = os.path.join(metadata.eval_output_dir, 'oracle_guided_v2_logs')
    os.makedirs(guided_log_dir, exist_ok=True)
    guided_log_path = os.path.join(guided_log_dir, f'{instance.instance_id}.jsonl')
    with open(guided_log_path, 'w', encoding='utf-8') as f:
        for entry in guided_log:
            f.write(json.dumps(entry) + '\n')

    test_result = {'git_patch': git_patch, 'oracle_guided_v2_log': guided_log}
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


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def _run_single_with_retries(
    instance: pd.Series,
    metadata: EvalMetadata,
    run_id: int,
    max_retries: int,
    timeout_seconds: int | None,
    executed_sids: list[str] | None = None,
) -> EvalOutput:
    runtime_failure_count = 0
    instance_id = str(instance.instance_id)

    for attempt in range(max_retries + 1):
        sid = _make_eval_sid(instance_id, run_id, attempt)
        if executed_sids is not None and sid not in executed_sids:
            executed_sids.append(sid)
        try:
            if timeout_seconds is not None:
                with timeout(timeout_seconds):
                    return _process_oracle_guided_v2_instance_with_sid(
                        instance, metadata, sid=sid,
                        reset_logger=True, runtime_failure_count=runtime_failure_count,
                    )
            return _process_oracle_guided_v2_instance_with_sid(
                instance, metadata, sid=sid,
                reset_logger=True, runtime_failure_count=runtime_failure_count,
            )
        except EvalTimeoutException:
            logger.exception(f'Timeout for instance {instance_id}')
            return EvalOutput(
                instance_id=instance.instance_id,
                test_result={},
                error=f'Timeout after {timeout_seconds} seconds',
            )
        except Exception as e:
            error = str(e)
            if attempt == max_retries:
                skip_errors = os.environ.get('EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED', 'false').lower() == 'true'
                if skip_errors:
                    return log_skipped_maximum_retries_exceeded(instance, metadata, e, max_retries)
                logger.exception(e)
                raise RuntimeError(f'Max retries for {instance_id}') from e

            if is_fatal_runtime_error(type(e).__name__ + ': ' + error):
                runtime_failure_count += 1
            logger.error(f'Retry {attempt+1}/{max_retries} for {instance_id}: {error}')
            time.sleep(5)

    raise RuntimeError('unreachable')


# ---------------------------------------------------------------------------
# Worker init + per-instance-all-runs loop
# ---------------------------------------------------------------------------

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
    global _WORKER_N_RUNS, _WORKER_SKIP_RUNS, _WORKER_RUN_CTX
    global _WORKER_RUNTIME, _WORKER_TIMEOUT_SECONDS, _WORKER_MAX_RETRIES
    global _WORKER_DATASET_TYPE, _OUTPUT_LOCKS, _PROGRESS_QUEUE

    _WORKER_N_RUNS = n_runs
    _WORKER_SKIP_RUNS = skip_runs
    _WORKER_RUN_CTX = run_ctx
    _WORKER_RUNTIME = runtime
    _WORKER_TIMEOUT_SECONDS = timeout_seconds
    _WORKER_MAX_RETRIES = max_retries
    _WORKER_DATASET_TYPE = dataset_type
    _OUTPUT_LOCKS = output_locks
    _PROGRESS_QUEUE = progress_queue
    base.DATASET_TYPE = dataset_type


def _append_result(output_file: str, result: EvalOutput, run_id: int):
    lock = _OUTPUT_LOCKS[run_id]
    with lock:
        with open(output_file, 'a') as f:
            f.write(result.model_dump_json() + '\n')
            f.flush()


def _is_max_retry_skipped(result: EvalOutput) -> bool:
    return bool(result.error) and str(result.error).lower().startswith('maximum retries')


def _process_instance_all_runs(instance_dict: dict[str, Any]) -> dict[str, Any]:
    """Process one instance across all runs sequentially (instance-major order)."""
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
        if instance_id in set(ctx['completed_ids']):
            skipped_runs.append(run_id)
            continue

        metadata = EvalMetadata.model_validate_json(ctx['metadata_json'])
        output_file = ctx['output_file']
        run_output_dir = ctx['run_output_dir']
        infer_log_file = os.path.join(
            run_output_dir, 'infer_logs', f'instance_{instance_id}.log'
        )

        if _PROGRESS_QUEUE is not None:
            _PROGRESS_QUEUE.put({
                'type': 'run_started',
                'instance_id': instance_id,
                'run_id': run_id,
                'infer_log_file': infer_log_file,
            })

        result = _run_single_with_retries(
            instance=instance,
            metadata=metadata,
            run_id=run_id,
            max_retries=_WORKER_MAX_RETRIES,
            timeout_seconds=_WORKER_TIMEOUT_SECONDS,
            executed_sids=executed_sids,
        )
        _append_result(output_file, result, run_id)
        processed_runs.append(run_id)

        status = 'skipped' if _is_max_retry_skipped(result) else (
            'failed' if bool(result.error) else 'finished'
        )
        if _PROGRESS_QUEUE is not None:
            _PROGRESS_QUEUE.put({
                'type': 'run_finished',
                'instance_id': instance_id,
                'run_id': run_id,
                'error': bool(result.error),
                'status': status,
                'test_result_preview': str(result.test_result)[:300],
            })

        if status == 'skipped':
            logger.info(
                f'### SKIPPED RUN {run_id} FOR INSTANCE {instance_id} '
                f'(maximum retries exceeded) ###'
            )
        else:
            logger.info(
                f'### FINISHED RUN {run_id} FOR INSTANCE {instance_id} '
                f'(error={bool(result.error)}) ###'
            )

    # Cleanup docker artifacts for this instance
    if _WORKER_RUNTIME == 'docker' and (processed_runs or skipped_runs):
        use_swebench_image = _WORKER_DATASET_TYPE != 'SWE-Gym'
        base_image = base.get_instance_docker_image(
            instance_id, swebench_official_image=use_swebench_image,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _ensure_open_file_limit(min_soft: int = 65535) -> None:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(max(soft, min_soft), hard) if hard != resource.RLIM_INFINITY else max(soft, min_soft)
        if target > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception as e:
        logger.warning(f'Failed to adjust RLIMIT_NOFILE: {e}')


if __name__ == '__main__':
    _ensure_open_file_limit()

    parser = get_evaluation_parser()
    parser.add_argument('--dataset', type=str, default='SWE-Gym/SWE-Gym')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--mode', type=str, default='swe', choices=['swe', 'swt', 'swt-ci'])
    parser.add_argument('--n-runs', type=int, default=1)
    parser.add_argument('--instance-ids', type=str, nargs='+', default=None)

    args, _ = parser.parse_known_args()

    if not args.agent_cls or args.agent_cls == 'CodeActAgent':
        args.agent_cls = 'OracleGuidedV2CodeActAgent'
        logger.info('Using default agent class: OracleGuidedV2CodeActAgent')

    # Execution salt for unique SIDs
    if not os.environ.get('OH_EVAL_EXECUTION_SALT'):
        os.environ['OH_EVAL_EXECUTION_SALT'] = hashlib.sha1(
            f'{time.time_ns()}-{os.getpid()}'.encode()
        ).hexdigest()[:10]
    logger.info(
        'Using OH_EVAL_EXECUTION_SALT=%s for SID namespacing.',
        os.environ['OH_EVAL_EXECUTION_SALT'],
    )

    dataset = load_dataset(args.dataset, split=args.split)
    base.set_dataset_type(args.dataset)
    swe_bench_tests = base.filter_dataset(dataset.to_pandas(), 'instance_id')

    if args.instance_ids:
        swe_bench_tests = swe_bench_tests[swe_bench_tests['instance_id'].isin(args.instance_ids)]
        missing = set(args.instance_ids) - set(swe_bench_tests['instance_id'])
        if missing:
            logger.warning(f'Instance IDs not found: {missing}')

    # Skip instances without preprocessed fact JSON files early
    preprocess_dir = os.environ.get('ORACLE_PREPROCESS_DIR', '').strip()
    if preprocess_dir:
        before_count = len(swe_bench_tests)
        has_facts = swe_bench_tests['instance_id'].apply(
            lambda iid: os.path.isfile(
                os.path.join(preprocess_dir, str(iid), 'stage2_facts.json')
            )
        )
        skipped = swe_bench_tests[~has_facts]['instance_id'].tolist()
        if skipped:
            logger.warning(
                f'Skipping {len(skipped)} instances without fact JSON: '
                f'{skipped[:10]}{"..." if len(skipped) > 10 else ""}'
            )
        swe_bench_tests = swe_bench_tests[has_facts]
        logger.info(f'Fact JSON filter: {before_count} → {len(swe_bench_tests)} instances')

    # Skip instances listed in the graph complexity filter JSON
    graph_filter_path = os.environ.get('ORACLE_GRAPH_FILTER_JSON', '').strip()
    if graph_filter_path and os.path.isfile(graph_filter_path):
        with open(graph_filter_path, 'r') as _gf:
            _graph_filter = json.load(_gf)
        _filtered_ids = set(_graph_filter.get('filtered_instance_ids', []))
        before_count = len(swe_bench_tests)
        swe_bench_tests = swe_bench_tests[
            ~swe_bench_tests['instance_id'].isin(_filtered_ids)
        ]
        logger.info(
            f'Graph complexity filter: {before_count} → {len(swe_bench_tests)} instances '
            f'(removed {before_count - len(swe_bench_tests)})'
        )
    elif graph_filter_path:
        logger.warning(f'ORACLE_GRAPH_FILTER_JSON not found: {graph_filter_path}')

    logger.info(f'Loaded {args.dataset} ({args.split}): {len(swe_bench_tests)} tasks')

    # Filter for SWE-Gym verified instances
    if base.DATASET_TYPE == 'SWE-Gym':
        verified_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'split', 'swegym_verified_instances.json',
        )
        if os.path.isfile(verified_path):
            with open(verified_path, 'r') as f:
                verified_ids = json.load(f)
            swe_bench_tests = swe_bench_tests[
                swe_bench_tests['instance_id'].isin(verified_ids)
            ]
            logger.info(f'{len(swe_bench_tests)} tasks after SWE-Gym verified filter')

    # Coerce PASS_TO_PASS / FAIL_TO_PASS to strings if needed
    if len(swe_bench_tests) > 0 and not isinstance(
        swe_bench_tests['PASS_TO_PASS'][swe_bench_tests['PASS_TO_PASS'].index[0]], str
    ):
        for col in ['PASS_TO_PASS', 'FAIL_TO_PASS']:
            swe_bench_tests[col] = swe_bench_tests[col].apply(lambda x: str(x))

    # Load GuidedConfigV2 early
    from openhands.agenthub.oracle_guided_v2_codeact_agent.guided_config_v2 import GuidedConfigV2
    _cfg = GuidedConfigV2.load()
    _cfg.export_to_env()

    llm_config = get_llm_config_arg(args.llm_config, args.config_file)
    if llm_config is None:
        raise ValueError(f'No LLM config: {args.llm_config}')
    llm_config.log_completions = True
    llm_config.modify_params = False

    condenser_name = os.environ.get('EVAL_CONDENSER')
    condenser_config = (
        get_condenser_config_arg(condenser_name, args.config_file) if condenser_name else None
    ) or NoOpCondenserConfig()

    agent_config = get_agent_config_arg(args.agent_config, args.config_file) if args.agent_config else None

    details = {'mode': args.mode}
    openhands.agenthub.Agent.get_cls(args.agent_cls)

    dataset_description = args.dataset.replace('/', '__') + '-' + args.split.replace('/', '__')

    n_runs = args.n_runs
    skip_runs_str = os.environ.get('SKIP_RUNS', '')
    skip_runs = set(int(x.strip()) for x in skip_runs_str.split(',') if x.strip())
    active_runs = [r for r in range(1, n_runs + 1) if r not in skip_runs]

    if not active_runs:
        logger.warning('All runs skipped. Nothing to do.')
        raise SystemExit(0)

    # Build per-run context
    run_context: dict[int, dict[str, Any]] = {}
    union_pending_ids: set[str] = set()

    for run_id in active_runs:
        run_eval_note = args.eval_note if n_runs == 1 else f'{args.eval_note}-run_{run_id}'
        run_metadata = make_metadata(
            llm_config, dataset_description, args.agent_cls,
            args.max_iterations, run_eval_note, args.eval_output_dir,
            details=details, agent_config=agent_config, condenser_config=condenser_config,
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

    # Build instance task list
    instance_tasks = [
        row.to_dict()
        for _, row in swe_bench_tests.iterrows()
        if str(row['instance_id']) in union_pending_ids
    ]

    total_instances = len(instance_tasks)
    total_pending = sum(run_context[r]['pending_count'] for r in active_runs)

    _GUIDED_V2_VARS = [
        'GUIDED_V2_NUM_CANDIDATES',
        'GUIDED_V2_MAX_RETRIES',
        'GUIDED_V2_PLANNER_HISTORY_NEAR_WINDOW',
        'GUIDED_V2_PLANNER_LLM_CONFIG',
        'GUIDED_V2_LEAKAGE_CRITIC_LLM_CONFIG',
        'GUIDED_V2_SUFFICIENCY_CRITIC_LLM_CONFIG',
        'GUIDED_V2_SAVE_PLANNER_PROMPTS',
        'GUIDED_V2_SAVE_CRITIC_PROMPTS',
        'GUIDED_V2_PLANNER_JSON_PARSE_MAX_RETRIES',
        'GUIDED_V2_LEAKAGE_CRITIC_JSON_PARSE_MAX_RETRIES',
        'GUIDED_V2_SUFFICIENCY_CRITIC_JSON_PARSE_MAX_RETRIES',
    ]

    logger.info('=' * 80)
    logger.info('ORACLE GUIDED V2 EVALUATION PLAN (INSTANCE-MAJOR):')
    logger.info(f'  Total instances in dataset:  {len(swe_bench_tests)}')
    logger.info(f'  Instances with pending runs: {total_instances}')
    logger.info(f'  Runs per instance:           {n_runs}')
    logger.info(f'  Active runs:                 {len(active_runs)} (skipping: {skip_runs if skip_runs else "none"})')
    logger.info(f'  Worker processes:            {args.eval_num_workers}')
    for run_id in active_runs:
        ctx = run_context[run_id]
        logger.info(f'  Run {run_id}: completed={len(ctx["completed_ids"])}, pending={ctx["pending_count"]}')
    logger.info(f'  Total pending evaluations:   {total_pending}')
    logger.info(f'  Guided V2 env var overrides (unset = YAML/default):')
    _any_set = False
    for _v in _GUIDED_V2_VARS:
        val = os.environ.get(_v)
        if val is not None:
            logger.info(f'    {_v}={val}')
            _any_set = True
    if not _any_set:
        logger.info('    (none — using YAML config / Python defaults)')
    logger.info('=' * 80)

    if total_instances == 0:
        logger.info('No pending instances. Done.')
        raise SystemExit(0)

    manager = Manager()
    output_locks = {run_id: manager.Lock() for run_id in active_runs}
    progress_queue = manager.Queue()

    start = time.time()
    completed_instances = 0

    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f'{int(seconds)}s'
        if seconds < 3600:
            return f'{int(seconds // 60)}m {int(seconds % 60)}s'
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f'{hours}h {minutes}m'

    with Pool(
        processes=args.eval_num_workers,
        initializer=_init_worker,
        initargs=(
            n_runs, skip_runs, run_context,
            os.environ.get('RUNTIME', 'docker'),
            8 * 60 * 60,  # 8h timeout per instance
            5,  # max retries
            base.DATASET_TYPE,
            output_locks,
            progress_queue,
        ),
    ) as pool:
        async_results = [
            pool.apply_async(_process_instance_all_runs, (task,))
            for task in instance_tasks
        ]

        with tqdm(total=total_pending, desc='Oracle-Guided V2 evaluations') as pbar:
            pending = list(async_results)
            while pending:
                drained_any = False

                # Drain progress queue
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

                # Check for completed workers
                still_pending = []
                for ar in pending:
                    if ar.ready():
                        try:
                            summary = ar.get()
                            completed_instances += 1
                            elapsed = time.time() - start
                            avg = elapsed / completed_instances
                            eta = avg * (total_instances - completed_instances)
                            logger.info(
                                f"Completed instance {summary['instance_id']} | "
                                f'instance progress {completed_instances}/{total_instances} | '
                                f'elapsed {_format_time(elapsed)} | eta {_format_time(eta)}'
                            )
                            logger.info(
                                f"### FINISHED INSTANCE {summary['instance_id']} | "
                                f"processed_runs={summary['processed_runs']} "
                                f"skipped_runs={summary['skipped_runs']} ###"
                            )
                        except Exception as e:
                            logger.error(f'Worker failed: {e}')
                    else:
                        still_pending.append(ar)
                pending = still_pending

                if pending and not drained_any:
                    time.sleep(0.5)

            # Final drain
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

    elapsed = time.time() - start
    logger.info('')
    logger.info('╔═══════════════════════════════════════════════════════════════════════════╗')
    logger.info('║ ALL ORACLE-GUIDED V2 EVALUATIONS COMPLETE!')
    logger.info(
        f'║ Processed instances: {completed_instances}/{total_instances} '
        f'| Total time: {_format_time(elapsed)}'
    )
    logger.info('╚═══════════════════════════════════════════════════════════════════════════╝')
    logger.info('')

    # Check for maximum-retries-exceeded markers
    for run_id in active_runs:
        eval_output_dir = os.path.dirname(run_context[run_id]['output_file'])
        check_maximum_retries_exceeded(eval_output_dir)
