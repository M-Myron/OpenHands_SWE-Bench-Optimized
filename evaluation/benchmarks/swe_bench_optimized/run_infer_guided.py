"""Entry point for the Guided Trajectory experiment on SWE-Bench.

This mirrors ``run_infer.py`` but uses :class:`GuidedCodeActAgent` which
embeds a Blinded Critic validation loop at every step.

Key differences from ``run_infer.py``:
  - Default agent class is ``GuidedCodeActAgent``.
  - Default instruction template is ``swe_guided.j2``
    (includes golden patch and test as a ``<REFERENCE_INFORMATION>`` block).
  - ``process_instance_guided`` wraps the standard ``process_instance``,
    clears the per-process :mod:`blinded_critic` log before each run, and
    attaches the validation log to ``EvalOutput.test_result``.

Configuration
-------------
Add a ``[llm.blinded_critic]`` section to ``config.toml``::

    [llm.blinded_critic]
    model = "gpt-4o-mini"
    api_key = "..."
    temperature = 0.0

The critic config name can be overridden via the env var
``BLINDED_CRITIC_LLM_CONFIG``.  If the config section is absent the agent
runs normally without validation.
"""

import json
import os
import time

import pandas as pd

# Register the guided agent BEFORE importing anything that resolves agent classes
import openhands.agenthub  # noqa: F401 — registers built-in agents
import openhands.agenthub.guided_codeact_agent  # noqa: F401 — registers GuidedCodeActAgent

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

# Register GuidedCodeActAgent in the fake-user-response dispatch table so that
# run_infer.process_instance can find it (it uses metadata.agent_class as key).
AGENT_CLS_TO_FAKE_USER_RESPONSE_FN['GuidedCodeActAgent'] = (
    AGENT_CLS_TO_FAKE_USER_RESPONSE_FN['CodeActAgent']
)
from openhands.agenthub.guided_codeact_agent.guided_codeact_agent import (
    clear_validation_log,
    read_and_clear_validation_log,
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
# Env-var defaults for guided mode
# ---------------------------------------------------------------------------

# Force the instruction template to the guided variant unless the caller
# overrides INSTRUCTION_TEMPLATE_NAME explicitly.
if not os.environ.get('INSTRUCTION_TEMPLATE_NAME'):
    os.environ['INSTRUCTION_TEMPLATE_NAME'] = 'swe_guided.j2'


# ---------------------------------------------------------------------------
# Guided process_instance wrapper
# ---------------------------------------------------------------------------


def process_instance_guided(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
    runtime_failure_count: int = 0,
) -> EvalOutput:
    """Wraps the standard :func:`process_instance` with validation-log capture.

    Steps:
    1. Clear the per-process Blinded Critic validation log.
    2. Run the standard ``process_instance`` (which now uses
       ``GuidedCodeActAgent`` if ``metadata.agent_class`` is set correctly).
    3. Read back the validation log written by the agent's ``step()`` method.
    4. Attach it to ``output.test_result['validation_log']``.
    """
    # Clear any leftover log from a previous instance in this worker process
    clear_validation_log()

    # If prompt-saving is enabled, configure a per-instance directory so each
    # instance's critic prompts land in their own folder and don't collide
    # across parallel workers.  The directory is cleared from the env after the
    # run so the next instance starts clean.
    _save_prompts = os.environ.get('BLINDED_CRITIC_SAVE_PROMPTS', '0').strip() == '1'
    if _save_prompts:
        prompts_dir = os.path.join(
            metadata.eval_output_dir,
            'blinded_critic_prompts',
            str(instance.instance_id),
        )
        os.environ['BLINDED_CRITIC_SAVE_PROMPTS_DIR'] = prompts_dir
        logger.info(f'[Guided] Critic prompt saving enabled → {prompts_dir}')
    else:
        # Ensure any leftover value from a previous instance is cleared
        os.environ.pop('BLINDED_CRITIC_SAVE_PROMPTS_DIR', None)

    output: EvalOutput = process_instance(
        instance=instance,
        metadata=metadata,
        reset_logger=reset_logger,
        runtime_failure_count=runtime_failure_count,
    )

    # Clean up the per-instance prompt-save dir env var so it doesn't bleed
    # into any post-processing code in this worker.
    os.environ.pop('BLINDED_CRITIC_SAVE_PROMPTS_DIR', None)

    # Collect what the Blinded Critic recorded during this run
    validation_log = read_and_clear_validation_log()

    # Write a dedicated per-instance Blinded Critic log for easy inspection
    critic_log_dir = os.path.join(metadata.eval_output_dir, 'blinded_critic_logs')
    os.makedirs(critic_log_dir, exist_ok=True)
    critic_log_path = os.path.join(critic_log_dir, f'{instance.instance_id}.jsonl')
    with open(critic_log_path, 'w') as f:
        for entry in validation_log:
            f.write(json.dumps(entry) + '\n')
    logger.info(f'[Guided] Critic log written to: {critic_log_path}')

    if validation_log:
        logger.info(
            f'[Guided] Instance {instance.instance_id}: '
            f'{len(validation_log)} critic validation entries recorded.'
        )
        # Count rejections for a quick summary
        rejections = sum(1 for e in validation_log if not e.get('valid', True))
        logger.info(
            f'[Guided] Critic summary: '
            f'{len(validation_log) - rejections} passed, {rejections} rejected.'
        )
    else:
        logger.info(
            f'[Guided] Instance {instance.instance_id}: no critic validations recorded '
            '(critic may be disabled or all steps were first-attempt valid).'
        )

    # Attach to test_result (backward-compatible: existing eval scripts ignore unknown keys)
    if output.test_result is None:
        output.test_result = {}
    output.test_result['validation_log'] = validation_log

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
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
        help='Dataset split to evaluate on.',
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

    # Default agent class to GuidedCodeActAgent
    if not args.agent_cls or args.agent_cls == 'CodeActAgent':
        args.agent_cls = 'GuidedCodeActAgent'
        logger.info(
            'Using default agent class: GuidedCodeActAgent'
        )

    dataset = load_dataset(args.dataset, split=args.split)
    set_dataset_type(args.dataset)

    swe_bench_tests = filter_dataset(dataset.to_pandas(), 'instance_id')
    logger.info(
        f'Loaded dataset {args.dataset} ({args.split}): {len(swe_bench_tests)} tasks'
    )

    # LLM config
    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config, args.config_file)
        if llm_config:
            llm_config.log_completions = True
            llm_config.modify_params = False

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm-config {args.llm_config}')

    # Condenser config
    condenser_name = os.environ.get('EVAL_CONDENSER')
    if condenser_name:
        condenser_config = get_condenser_config_arg(condenser_name, args.config_file)
        if condenser_config is None:
            logger.warning(
                f'Could not find condenser config: {condenser_name}. '
                'Using NoOpCondenser.'
            )
            condenser_config = NoOpCondenserConfig()
    else:
        condenser_config = NoOpCondenserConfig()

    # Agent config
    agent_config = None
    if args.agent_config:
        agent_config = get_agent_config_arg(args.agent_config, args.config_file)

    details = {'mode': args.mode}
    import openhands.agenthub as _hub

    _agent_cls = _hub.Agent.get_cls(args.agent_cls)

    dataset_description = (
        args.dataset.replace('/', '__') + '-' + args.split.replace('/', '__')
    )

    # Multi-run / batching logic (mirrors run_infer.py)
    n_runs = args.n_runs
    skip_runs_str = os.environ.get('SKIP_RUNS', '')
    skip_runs = set(int(x.strip()) for x in skip_runs_str.split(',') if x.strip())
    total_runs = n_runs - len(skip_runs)

    batch_size = args.eval_num_workers
    total_instances = len(swe_bench_tests)
    total_batches = (total_instances + batch_size - 1) // batch_size

    logger.info('=' * 80)
    logger.info('GUIDED TRAJECTORY EVALUATION PLAN:')
    logger.info(f'  Dataset:              {args.dataset}')
    logger.info(f'  Agent class:          {args.agent_cls}')
    logger.info(f'  Instruction template: {os.environ.get("INSTRUCTION_TEMPLATE_NAME")}')
    logger.info(f'  Blinded Critic LLM:   {os.environ.get("BLINDED_CRITIC_LLM_CONFIG", "blinded_critic")}')
    logger.info(f'  Total instances:      {total_instances}')
    logger.info(f'  Batch size:           {batch_size}')
    logger.info(f'  Total batches:        {total_batches}')
    logger.info(f'  Runs per instance:    {n_runs}')
    logger.info(f'  Active runs:          {total_runs}')
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
            run_output_file = os.path.join(
                run_metadata.eval_output_dir, 'output.jsonl'
            )

            if batch_start == 0:
                print(f'### OUTPUT FILE FOR RUN {run_id}: {run_output_file} ###')

            instances = prepare_dataset(
                batch_instances, run_output_file, args.eval_n_limit
            )

            if len(instances) == 0:
                logger.info(
                    f'All instances in batch {batch_num}, run {run_id} already done.'
                )
                continue

            # Convert list columns if needed
            for col in ['PASS_TO_PASS', 'FAIL_TO_PASS']:
                if col in instances.columns and len(instances) > 0:
                    first_val = instances[col].iloc[0]
                    if not isinstance(first_val, str):
                        instances[col] = instances[col].apply(str)

            num_workers_for_batch = min(len(instances), args.eval_num_workers)

            run_evaluation(
                instances,
                run_metadata,
                run_output_file,
                num_workers_for_batch,
                process_instance_guided,
                timeout_seconds=8 * 60 * 60,
                max_retries=5,
            )

            completed_evaluations += 1
            logger.info(
                f'Completed run {run_id}/{n_runs} for batch {batch_num}/{total_batches}'
            )

    total_time = time.time() - start_time
    hours, rem = divmod(int(total_time), 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info('')
    logger.info('ALL GUIDED EVALUATIONS COMPLETE!')
    logger.info(
        f'Total time: {hours}h {minutes}m {seconds}s '
        f'| {completed_evaluations} evaluations'
    )
