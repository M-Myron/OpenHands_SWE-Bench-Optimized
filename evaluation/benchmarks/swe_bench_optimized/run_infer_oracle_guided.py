"""Evaluation entry point for the Oracle Guided CodeAct Agent.

Handles per-instance oracle context preparation, fact loading, prompt saving,
and guided log collection.
"""

import json
import os
import time

import pandas as pd

import openhands.agenthub  # noqa: F401
import openhands.agenthub.oracle_guided_codeact_agent  # noqa: F401
from datasets import load_dataset

from evaluation.benchmarks.swe_bench_optimized.run_infer import (
    AGENT_CLS_TO_FAKE_USER_RESPONSE_FN,
    filter_dataset,
    process_instance,
    set_dataset_type,
)
from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    make_metadata,
    prepare_dataset,
    run_evaluation,
)
from openhands.agenthub.oracle_guided_codeact_agent.oracle_guided_codeact_agent import (
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
from openhands.core.logger import openhands_logger as logger


# ---------------------------------------------------------------------------
# Register fake-user-response for this agent
# ---------------------------------------------------------------------------
AGENT_CLS_TO_FAKE_USER_RESPONSE_FN['OracleGuidedCodeActAgent'] = (
    AGENT_CLS_TO_FAKE_USER_RESPONSE_FN.get('CodeActAgent')
)


# ---------------------------------------------------------------------------
# Oracle context helpers
# ---------------------------------------------------------------------------

def _build_issue_understanding(instance: pd.Series) -> str:
    """Assemble issue understanding from dataset columns."""
    parts: list[str] = []
    for col in ('issue_understanding', 'bug_description', 'bug_trigger',
                'fix_rationale', 'hints_text'):
        val = getattr(instance, col, None) if hasattr(instance, col) else instance.get(col)
        if val and str(val).strip():
            parts.append(f'### {col.replace("_", " ").title()}\n{val}')
    return '\n\n'.join(parts) if parts else '(No structured issue understanding available.)'


def _load_react_facts(instance_id: str) -> dict | None:
    """Load v6 structured facts from stage2_facts.json."""
    preprocess_dir = os.environ.get('ORACLE_PREPROCESS_DIR', '').strip()
    if not preprocess_dir:
        return None

    # v6 layout: {preprocess_dir}/{instance_id}/stage2_facts.json
    facts_path = os.path.join(preprocess_dir, instance_id, 'stage2_facts.json')
    if os.path.isfile(facts_path):
        try:
            with open(facts_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f'[OracleGuided] Loaded v6 facts from {facts_path}')
            return data
        except Exception as exc:
            logger.warning(f'[OracleGuided] Failed to read facts: {exc}')
            return None

    logger.info(f'[OracleGuided] No react facts found for {instance_id}')
    return None


def _write_oracle_context_file(instance: pd.Series, metadata: EvalMetadata) -> str:
    """Write oracle context JSON for a single instance."""
    context_dir = os.path.join(metadata.eval_output_dir, 'oracle_guided_context')
    os.makedirs(context_dir, exist_ok=True)

    react_facts = _load_react_facts(str(instance.instance_id))

    payload = {
        'instance_id': str(instance.instance_id),
        'patch': str(instance.patch),
        'test_patch': str(instance.test_patch),
        'issue_understanding': _build_issue_understanding(instance),
        'react_facts': react_facts,
    }

    path = os.path.join(context_dir, f'{instance.instance_id}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    return path


# ---------------------------------------------------------------------------
# Instance wrapper
# ---------------------------------------------------------------------------

def process_instance_oracle_guided(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
    runtime_failure_count: int = 0,
) -> EvalOutput:
    """Process a single instance with Oracle Guided agent."""
    clear_triage_log()

    # Write oracle context
    context_path = _write_oracle_context_file(instance, metadata)
    os.environ['ORACLE_GUIDED_CONTEXT_PATH'] = context_path

    # Set prompt saving directories
    save_planner = os.environ.get('GUIDED_SAVE_PLANNER_PROMPTS', '1').strip() == '1'
    if save_planner:
        planner_dir = os.path.join(
            metadata.eval_output_dir,
            'oracle_guided_planner_prompts',
            str(instance.instance_id),
        )
        os.environ['GUIDED_PLANNER_SAVE_PROMPTS_DIR'] = planner_dir
    else:
        os.environ.pop('GUIDED_PLANNER_SAVE_PROMPTS_DIR', None)

    save_critic = os.environ.get('GUIDED_SAVE_CRITIC_PROMPTS', '1').strip() == '1'
    if save_critic:
        critic_dir = os.path.join(
            metadata.eval_output_dir,
            'oracle_guided_critic_prompts',
            str(instance.instance_id),
        )
        os.environ['GUIDED_CRITIC_SAVE_PROMPTS_DIR'] = critic_dir
    else:
        os.environ.pop('GUIDED_CRITIC_SAVE_PROMPTS_DIR', None)

    try:
        output: EvalOutput = process_instance(
            instance=instance,
            metadata=metadata,
            reset_logger=reset_logger,
            runtime_failure_count=runtime_failure_count,
        )
    finally:
        os.environ.pop('ORACLE_GUIDED_CONTEXT_PATH', None)
        os.environ.pop('GUIDED_PLANNER_SAVE_PROMPTS_DIR', None)
        os.environ.pop('GUIDED_CRITIC_SAVE_PROMPTS_DIR', None)

    # Collect guided logs
    guided_log = read_and_clear_triage_log()
    log_dir = os.path.join(metadata.eval_output_dir, 'oracle_guided_logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{instance.instance_id}.jsonl')

    with open(log_path, 'w', encoding='utf-8') as f:
        for entry in guided_log:
            f.write(json.dumps(entry) + '\n')

    logger.info(f'[OracleGuided] Log written: {log_path}')

    if output.test_result is None:
        output.test_result = {}
    output.test_result['oracle_guided_log'] = guided_log

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = get_evaluation_parser()
    parser.add_argument(
        '--dataset', type=str, default='SWE-Gym/SWE-Gym',
        help='Dataset to evaluate on.',
    )
    parser.add_argument(
        '--split', type=str, default='train',
        help='Dataset split to evaluate on.',
    )
    parser.add_argument(
        '--mode', type=str, default='swe', choices=['swe', 'swt', 'swt-ci'],
        help='Evaluation mode.',
    )
    parser.add_argument(
        '--n-runs', type=int, default=1,
        help='Number of runs per instance.',
    )
    parser.add_argument(
        '--instance-ids', type=str, nargs='+', default=None,
        help='Optional list of instance IDs to evaluate.',
    )

    args, _ = parser.parse_known_args()

    if not args.agent_cls or args.agent_cls == 'CodeActAgent':
        args.agent_cls = 'OracleGuidedCodeActAgent'
        logger.info('Using default agent class: OracleGuidedCodeActAgent')

    dataset = load_dataset(args.dataset, split=args.split)
    set_dataset_type(args.dataset)

    swe_bench_tests = filter_dataset(dataset.to_pandas(), 'instance_id')

    if args.instance_ids:
        swe_bench_tests = swe_bench_tests[
            swe_bench_tests['instance_id'].isin(args.instance_ids)
        ]
        missing = set(args.instance_ids) - set(swe_bench_tests['instance_id'])
        if missing:
            logger.warning(f'Instance IDs not found in dataset: {missing}')

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
            f'(removed {before_count - len(swe_bench_tests)} at p{_graph_filter.get("thresholds", {}).get("percentile", 95)})'
        )
    elif graph_filter_path:
        logger.warning(f'ORACLE_GRAPH_FILTER_JSON not found: {graph_filter_path}')

    logger.info(
        f'Loaded dataset {args.dataset} ({args.split}): {len(swe_bench_tests)} tasks'
    )

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config, args.config_file)
        if llm_config:
            llm_config.log_completions = True
            llm_config.modify_params = False

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm-config {args.llm_config}')

    condenser_name = os.environ.get('EVAL_CONDENSER')
    if condenser_name:
        condenser_config = get_condenser_config_arg(condenser_name, args.config_file)
        if condenser_config is None:
            logger.warning(
                f'Could not find condenser config: {condenser_name}. Using NoOpCondenser.'
            )
            condenser_config = NoOpCondenserConfig()
    else:
        condenser_config = NoOpCondenserConfig()

    agent_config = None
    if args.agent_config:
        agent_config = get_agent_config_arg(args.agent_config, args.config_file)

    details = {'mode': args.mode}
    import openhands.agenthub as _hub
    _hub.Agent.get_cls(args.agent_cls)

    dataset_description = (
        args.dataset.replace('/', '__') + '-' + args.split.replace('/', '__')
    )

    n_runs = args.n_runs
    skip_runs_str = os.environ.get('SKIP_RUNS', '')
    skip_runs = set(int(x.strip()) for x in skip_runs_str.split(',') if x.strip())

    batch_size = args.eval_num_workers
    total_instances = len(swe_bench_tests)
    total_batches = (total_instances + batch_size - 1) // batch_size

    # Load GuidedConfig early
    from openhands.agenthub.oracle_guided_codeact_agent.guided_config import GuidedConfig
    _cfg = GuidedConfig.load()
    _cfg.export_to_env()

    logger.info('=' * 80)
    logger.info('ORACLE GUIDED EVALUATION PLAN:')
    logger.info(f'  Dataset:                       {args.dataset}')
    logger.info(f'  Agent class:                   {args.agent_cls}')
    logger.info(f'  Total instances:               {total_instances}')
    logger.info(f'  Batch size:                    {batch_size}')
    logger.info(f'  Total batches:                 {total_batches}')
    logger.info(f'  Runs per instance:             {n_runs}')
    logger.info(
        f'  Solver candidates:             {os.environ.get("GUIDED_NUM_CANDIDATES", "1")}'
    )
    logger.info(
        f'  Planner retries:               {os.environ.get("GUIDED_PLANNER_MAX_RETRIES", "2")}'
    )
    logger.info(
        f'  Planner LLM config:            {os.environ.get("GUIDED_PLANNER_LLM_CONFIG", "oracle_planner")}'
    )
    logger.info(
        f'  Critic LLM config:             {os.environ.get("GUIDED_CRITIC_LLM_CONFIG", "blinded_critic")}'
    )
    logger.info('=' * 80)

    start_time = time.time()
    completed_evaluations = 0

    for batch_start in range(0, total_instances, batch_size):
        batch_end = min(batch_start + batch_size, total_instances)
        batch_instances = swe_bench_tests.iloc[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1

        for run_id in range(1, n_runs + 1):
            if run_id in skip_runs:
                logger.info(f'Skipping run {run_id}/{n_runs} for batch {batch_num}')
                continue

            run_eval_note = (
                f'{args.eval_note}-run_{run_id}' if n_runs > 1 else args.eval_note
            )

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

            if batch_start == 0:
                print(f'### OUTPUT FILE FOR RUN {run_id}: {run_output_file} ###')

            instances = prepare_dataset(
                batch_instances, run_output_file, args.eval_n_limit
            )

            if len(instances) == 0:
                logger.info(
                    f'Batch {batch_num}/{total_batches} run {run_id}: '
                    'no remaining instances to evaluate.'
                )
                continue

            logger.info(
                f'Batch {batch_num}/{total_batches} run {run_id}/{n_runs}: '
                f'evaluating {len(instances)} instances...'
            )

            run_evaluation(
                instances,
                metadata=run_metadata,
                output_file=run_output_file,
                num_workers=args.eval_num_workers,
                process_instance_func=process_instance_oracle_guided,
            )

            completed_evaluations += len(instances)
            elapsed = time.time() - start_time
            logger.info(
                f'Completed: {completed_evaluations} evaluations in {elapsed / 60:.1f}m'
            )

    total_elapsed = time.time() - start_time
    logger.info(
        f'Oracle-Guided evaluation finished: {completed_evaluations} evaluations, '
        f'{total_elapsed / 60:.1f}m total.'
    )
