"""Instance-major inference using the original SWE-Gym paper's prompt/tool format.

Combines the instance-major scheduling from ``run_infer_instance_major.py``
(Pool-based, no batch boundaries, per-instance Docker cleanup) with the
SweGymLegacyCodeActAgent configuration from ``run_infer_swegym_legacy.py``
(original 3-tool format, simple system prompt).

Key advantages over ``run_infer_swegym_legacy.py``:
  - No batch boundaries: workers pick up the next instance immediately
  - Per-instance Docker cleanup: no background cleanup process needed
  - Better worker utilization: no idle workers waiting for batch stragglers

Usage:
    bash evaluation/benchmarks/swe_bench_optimized/scripts/run_infer_swegym_legacy_instance_major.sh \\
        llm.mymodel HEAD SweGymLegacyCodeActAgent 500 100 32
"""

import os

# Register agents
import openhands.agenthub  # noqa: F401 — registers built-in agents
import openhands.agenthub.swegym_legacy_codeact_agent  # noqa: F401 — registers SweGymLegacyCodeActAgent

import evaluation.benchmarks.swe_bench_optimized.run_infer as _run_infer_module
from evaluation.benchmarks.swe_bench_optimized.run_infer import (
    AGENT_CLS_TO_FAKE_USER_RESPONSE_FN,
    get_instance_docker_image,
)
from evaluation.benchmarks.swe_bench.resource.mapping import (
    get_instance_resource_factor,
)
from evaluation.utils.shared import (
    codeact_user_response,
    get_default_sandbox_config_for_eval,
    get_openhands_config_for_eval,
    update_llm_config_for_completions_logging,
)
from openhands.core.config import (
    AgentConfig,
    get_llm_config_arg,
    get_llms_for_routing_config,
    get_model_routing_config_arg,
)
from openhands.core.logger import openhands_logger as logger

# Register SweGymLegacyCodeActAgent in the dispatch table
AGENT_CLS_TO_FAKE_USER_RESPONSE_FN['SweGymLegacyCodeActAgent'] = codeact_user_response

# Force the instruction template to the legacy format unless overridden
if not os.environ.get('INSTRUCTION_TEMPLATE_NAME'):
    os.environ['INSTRUCTION_TEMPLATE_NAME'] = 'swe_swegym_legacy.j2'


def get_config_swegym_legacy(instance, metadata):
    """Create config using the legacy SWE-Gym tools and system prompt.

    Overrides the agent config to:
    - Use system_prompt_swegym_legacy.j2
    - Disable plan_mode (no task_tracker)
    - Disable think tool
    - Disable jupyter, browsing, mcp, prompt_extensions
    """
    use_swebench_official_image = _run_infer_module.DATASET_TYPE != 'SWE-Gym'
    base_container_image = get_instance_docker_image(
        instance['instance_id'],
        swebench_official_image=use_swebench_official_image,
    )
    logger.info(f'Using instance container image: {base_container_image}.')

    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = base_container_image
    sandbox_config.enable_auto_lint = True
    sandbox_config.use_host_network = False
    sandbox_config.platform = 'linux/amd64'
    sandbox_config.remote_runtime_resource_factor = get_instance_resource_factor(
        dataset_name=metadata.dataset,
        instance_id=instance['instance_id'],
    )

    config = get_openhands_config_for_eval(
        metadata=metadata,
        enable_browser=False,  # Always disabled for legacy
        runtime=os.environ.get('RUNTIME', 'docker'),
        sandbox_config=sandbox_config,
    )

    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config, metadata.eval_output_dir, instance['instance_id']
        )
    )
    config.set_llm_config(get_llm_config_arg('draft_editor'), 'draft_editor')

    model_routing_config = get_model_routing_config_arg()
    model_routing_config.llms_for_routing = get_llms_for_routing_config()

    # Legacy agent config: minimal features matching original SWE-Gym paper
    agent_config = AgentConfig(
        enable_jupyter=False,
        enable_browsing=False,
        enable_llm_editor=False,
        enable_mcp=False,
        enable_think=False,         # No think tool in original
        enable_plan_mode=False,     # No task_tracker in original
        enable_prompt_extensions=False,
        condenser=metadata.condenser_config,
        model_routing=model_routing_config,
        system_prompt_filename='system_prompt_swegym_legacy.j2',
    )
    config.set_agent_config(agent_config)

    return config


# Monkey-patch get_config so that the instance-major worker uses legacy config
_original_get_config = _run_infer_module.get_config


def _patched_get_config(instance, metadata):
    """Route to legacy config for SweGymLegacyCodeActAgent."""
    if metadata.agent_class == 'SweGymLegacyCodeActAgent':
        return get_config_swegym_legacy(instance, metadata)
    return _original_get_config(instance, metadata)


_run_infer_module.get_config = _patched_get_config


# ---------------------------------------------------------------------------
# Re-use the instance-major __main__ block by importing and running it.
# The monkey-patch above ensures the correct config is used for legacy agents.
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Import triggers the __main__ guard check. We need to run its code directly.
    # Instead, we replicate the main block from run_infer_instance_major.py
    # but with our monkey-patches already applied.

    import hashlib
    import json
    import queue
    import time
    from multiprocessing import Manager, Pool

    from datasets import load_dataset
    from tqdm import tqdm

    from evaluation.benchmarks.swe_bench_optimized.run_infer_instance_major import (
        _cleanup_instance_docker_artifacts,
        _ensure_open_file_limit,
        _format_time,
        _init_worker,
        _process_instance_all_runs,
    )
    from evaluation.utils.shared import (
        check_maximum_retries_exceeded,
        make_metadata,
        prepare_dataset,
    )
    from openhands.core.config import (
        get_agent_config_arg,
        get_evaluation_parser,
        get_llm_config_arg,
    )
    from openhands.core.config.condenser_config import NoOpCondenserConfig
    from openhands.core.config.utils import get_condenser_config_arg

    import evaluation.benchmarks.swe_bench_optimized.run_infer as base

    _ensure_open_file_limit()

    parser = get_evaluation_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='princeton-nlp/SWE-bench_Verified',
        help='Dataset to evaluate on.',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Dataset split.',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='swe',
        choices=['swe', 'swt', 'swt-ci'],
        help="Evaluation mode.",
    )
    parser.add_argument(
        '--n-runs',
        type=int,
        default=1,
        help='Number of runs per instance.',
    )

    args, _ = parser.parse_known_args()

    # Ensure each top-level rollout run uses a unique SID namespace
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

    run_context: dict[int, dict] = {}
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

    instance_tasks = []
    for _, row in swe_bench_tests.iterrows():
        if str(row['instance_id']) in union_pending_ids:
            instance_tasks.append(row.to_dict())

    total_instances_to_process = len(instance_tasks)
    logger.info('=' * 80)
    logger.info('SWEGYM LEGACY EVALUATION PLAN (INSTANCE-MAJOR):')
    logger.info(f'  Agent: {args.agent_cls} (original SWE-Gym paper format)')
    logger.info(f'  Dataset: {args.dataset} ({args.split})')
    logger.info(f'  System prompt: system_prompt_swegym_legacy.j2')
    logger.info(f'  Instruction template: {os.environ.get("INSTRUCTION_TEMPLATE_NAME", "swe_swegym_legacy.j2")}')
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

                next_pending = []
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

            # Drain any late queue events
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
    logger.info('║ ALL EVALUATIONS COMPLETE! (SweGym Legacy Instance-Major)')
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
