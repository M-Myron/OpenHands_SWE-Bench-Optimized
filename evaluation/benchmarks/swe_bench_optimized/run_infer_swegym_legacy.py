"""Entry point for inference/rollout using the original SWE-Gym paper's prompt/tool format.

Mirrors ``run_infer.py`` but uses ``SweGymLegacyCodeActAgent`` with the original
SWE-Gym paper's system prompt, tool definitions, and task instruction format.
Works with ANY dataset (SWE-bench, SWE-Gym, SWE-bench-Live, etc.).

Compatibility targets:
- Released SFT trajectories: https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories
- Original paper: https://arxiv.org/abs/2412.21139

Key differences from ``run_infer.py``:
  - Uses ``SweGymLegacyCodeActAgent`` with 3 simple legacy tools
    (execute_bash, finish, str_replace_editor — no security_risk, no think, no task_tracker)
  - System prompt is the original simple format (no ROLE/EFFICIENCY/etc. sections)
  - Default instruction template is ``swe_swegym_legacy.j2`` (simpler 5-step format)
  - Agent config has enable_plan_mode=False, enable_think=False, etc.

Usage (for SWE-bench evaluation):
    bash evaluation/benchmarks/swe_bench_optimized/scripts/run_infer_swegym_legacy.sh \\
        llm.mymodel HEAD SweGymLegacyCodeActAgent 500 100 16

Usage (for SWE-Gym rollout):
    bash evaluation/benchmarks/swe_bench_optimized/scripts/rollout_swegym_legacy.sh \\
        llm.mymodel 'train-legacy' 16
"""

import json
import os
import time

import pandas as pd

# Register agents
import openhands.agenthub  # noqa: F401 — registers built-in agents
import openhands.agenthub.swegym_legacy_codeact_agent  # noqa: F401 — registers SweGymLegacyCodeActAgent

from datasets import load_dataset

import evaluation.benchmarks.swe_bench_optimized.run_infer as _run_infer_module
from evaluation.benchmarks.swe_bench_optimized.run_infer import (
    AGENT_CLS_TO_FAKE_USER_RESPONSE_FN,
    _env_prepare_concurrency_slot,
    _get_swebench_workspace_dir_name,
    complete_runtime,
    filter_dataset,
    get_instance_docker_image,
    get_instruction,
    initialize_runtime,
    process_instance,
    set_dataset_type,
    RUN_WITH_BROWSING,
    ENABLE_LLM_EDITOR,
)
from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    codeact_user_response,
    check_maximum_retries_exceeded,
    get_default_sandbox_config_for_eval,
    get_openhands_config_for_eval,
    make_metadata,
    prepare_dataset,
    run_evaluation,
    update_llm_config_for_completions_logging,
)
from openhands.core.config import (
    AgentConfig,
    get_agent_config_arg,
    get_evaluation_parser,
    get_llm_config_arg,
    get_llms_for_routing_config,
    get_model_routing_config_arg,
)
from openhands.core.config.condenser_config import NoOpCondenserConfig
from openhands.core.config.utils import get_condenser_config_arg
from openhands.core.logger import openhands_logger as logger
from openhands.events.serialization.event import event_from_dict, event_to_dict
from openhands.critic import AgentFinishedCritic

from evaluation.benchmarks.swe_bench.resource.mapping import (
    get_instance_resource_factor,
)

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


# Monkey-patch the get_config used in process_instance
# This is the cleanest way to reuse process_instance without copying it
_original_get_config = _run_infer_module.get_config


def _patched_get_config(instance, metadata):
    """Route to legacy config for SweGymLegacyCodeActAgent."""
    if metadata.agent_class == 'SweGymLegacyCodeActAgent':
        return get_config_swegym_legacy(instance, metadata)
    return _original_get_config(instance, metadata)


_run_infer_module.get_config = _patched_get_config


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

    dataset = load_dataset(args.dataset, split=args.split)
    set_dataset_type(args.dataset)

    swe_bench_tests = filter_dataset(dataset.to_pandas(), 'instance_id')
    logger.info(
        f'Loaded dataset {args.dataset} with split {args.split}: {len(swe_bench_tests)} tasks'
    )

    # Handle SWE-Gym verified instances filter
    if _run_infer_module.DATASET_TYPE == 'SWE-Gym':
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

    ITERATIVE_EVAL_MODE = (
        os.environ.get('ITERATIVE_EVAL_MODE', 'false').lower() == 'true'
    )
    ITERATIVE_EVAL_MODE_MAX_ATTEMPTS = int(
        os.environ.get('ITERATIVE_EVAL_MODE_MAX_ATTEMPTS', '3')
    )

    if not ITERATIVE_EVAL_MODE:
        n_runs = args.n_runs
        skip_runs_str = os.environ.get('SKIP_RUNS', '')
        skip_runs = set(int(x.strip()) for x in skip_runs_str.split(',') if x.strip())

        batch_size = args.eval_num_workers
        total_instances = len(swe_bench_tests)
        total_batches = (total_instances + batch_size - 1) // batch_size
        total_runs = n_runs - len(skip_runs)

        logger.info(f'=' * 80)
        logger.info(f'SWEGYM LEGACY FORMAT EVALUATION PLAN:')
        logger.info(f'  Agent: {args.agent_cls} (original SWE-Gym paper format)')
        logger.info(f'  Dataset: {args.dataset} ({args.split})')
        logger.info(f'  System prompt: system_prompt_swegym_legacy.j2')
        logger.info(f'  Instruction template: {os.environ.get("INSTRUCTION_TEMPLATE_NAME", "swe_swegym_legacy.j2")}')
        logger.info(f'  Total instances: {total_instances}')
        logger.info(f'  Batch size (workers): {batch_size}')
        logger.info(f'  Runs per instance: {n_runs}')
        logger.info(f'  Active runs: {total_runs} (skipping: {skip_runs if skip_runs else "none"})')
        logger.info(f'=' * 80)

        start_time = time.time()
        completed_evaluations = 0
        evaluation_times = []

        for batch_start in range(0, total_instances, batch_size):
            batch_end = min(batch_start + batch_size, total_instances)
            batch_instances = swe_bench_tests.iloc[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1

            logger.info(f'BATCH {batch_num}/{total_batches}: instances {batch_start+1}-{batch_end}')

            for run_id in range(1, n_runs + 1):
                if run_id in skip_runs:
                    continue

                if n_runs > 1:
                    run_eval_note = f'{args.eval_note}-run_{run_id}'
                else:
                    run_eval_note = args.eval_note

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

                instances = prepare_dataset(batch_instances, run_output_file, args.eval_n_limit)

                if len(instances) == 0:
                    logger.info(f'  All instances in batch {batch_num}, run {run_id} already completed')
                    continue

                if len(instances) > 0 and not isinstance(
                    instances['PASS_TO_PASS'][instances['PASS_TO_PASS'].index[0]], str
                ):
                    for col in ['PASS_TO_PASS', 'FAIL_TO_PASS']:
                        instances[col] = instances[col].apply(lambda x: str(x))

                logger.info(f'  Evaluating {len(instances)} instances (batch {batch_num}, run {run_id})...')

                eval_start_time = time.time()
                num_workers_for_batch = min(len(instances), args.eval_num_workers)

                run_evaluation(
                    instances,
                    run_metadata,
                    run_output_file,
                    num_workers_for_batch,
                    process_instance,
                    timeout_seconds=8 * 60 * 60,
                    max_retries=5,
                )

                eval_time = time.time() - eval_start_time
                evaluation_times.append(eval_time)
                if len(evaluation_times) > 10:
                    evaluation_times.pop(0)
                completed_evaluations += 1

        logger.info(f'ALL EVALUATIONS COMPLETE! ({completed_evaluations} evaluations)')

    else:
        # ITERATIVE_EVAL_MODE
        metadata = make_metadata(
            llm_config,
            dataset_description,
            args.agent_cls,
            args.max_iterations,
            args.eval_note,
            args.eval_output_dir,
            details=details,
            agent_config=agent_config,
            condenser_config=condenser_config,
        )
        output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
        print(f'### OUTPUT FILE: {output_file} ###')

        critic = AgentFinishedCritic()

        def get_cur_output_file_path(attempt: int) -> str:
            return (
                f'{output_file.removesuffix(".jsonl")}.critic_attempt_{attempt}.jsonl'
            )

        eval_ids = None
        for attempt in range(1, ITERATIVE_EVAL_MODE_MAX_ATTEMPTS + 1):
            cur_output_file = get_cur_output_file_path(attempt)
            logger.info(
                f'Running evaluation with critic for attempt {attempt} of {ITERATIVE_EVAL_MODE_MAX_ATTEMPTS}.'
            )

            if attempt > 1 and metadata.llm_config.temperature == 0:
                metadata.llm_config.temperature = 0.1

            instances = prepare_dataset(
                swe_bench_tests, cur_output_file, args.eval_n_limit, eval_ids=eval_ids
            )
            if len(instances) > 0 and not isinstance(
                instances['PASS_TO_PASS'][instances['PASS_TO_PASS'].index[0]], str
            ):
                for col in ['PASS_TO_PASS', 'FAIL_TO_PASS']:
                    instances[col] = instances[col].apply(lambda x: str(x))

            run_evaluation(
                instances,
                metadata,
                cur_output_file,
                args.eval_num_workers,
                process_instance,
                timeout_seconds=8 * 60 * 60,
                max_retries=5,
            )

            instances_failed = []
            with open(cur_output_file, 'r') as f:
                for line in f:
                    instance = json.loads(line)
                    try:
                        history = [
                            event_from_dict(event) for event in instance['history']
                        ]
                        critic_result = critic.evaluate(
                            history, instance['test_result'].get('git_patch', '')
                        )
                        if not critic_result.success:
                            instances_failed.append(instance['instance_id'])
                    except Exception as e:
                        logger.error(
                            f'Error loading history for instance {instance["instance_id"]}: {e}'
                        )
                        instances_failed.append(instance['instance_id'])
            eval_ids = instances_failed

            if len(instances_failed) == 0:
                break

        # Aggregate results
        fout = open(output_file, 'w')
        added_instance_ids = set()
        for attempt in reversed(range(1, ITERATIVE_EVAL_MODE_MAX_ATTEMPTS + 1)):
            cur_output_file = get_cur_output_file_path(attempt)
            if not os.path.exists(cur_output_file):
                continue
            with open(cur_output_file, 'r') as f:
                for line in f:
                    instance = json.loads(line)
                    if (
                        instance['instance_id'] not in added_instance_ids
                        and instance['test_result'].get('git_patch', '').strip()
                    ):
                        fout.write(line)
                        added_instance_ids.add(instance['instance_id'])
        fout.close()
        logger.info(f'Done! Total {len(added_instance_ids)} instances added to {output_file}')
        check_maximum_retries_exceeded(metadata.eval_output_dir)
