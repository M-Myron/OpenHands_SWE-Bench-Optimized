"""
SWE-smith inference runner for OpenHands evaluation.

This script enables running OpenHands agents on SWE-smith instances
(specifically the real PR-mirror subset). It adapts SWE-smith's dataset
format to work with the existing OpenHands evaluation pipeline.

Usage:
    poetry run python evaluation/benchmarks/swe_bench_optimized/swe_smith/run_infer.py \
        --agent-cls CodeActAgent \
        --llm-config llm.eval \
        --max-iterations 100 \
        --eval-num-workers 4 \
        --dataset SWE-bench/SWE-smith \
        --split train \
        --n-runs 1

The output structure matches SWE-Gym format under:
    evaluation/evaluation_outputs/outputs/SWE-bench__SWE-smith-train/CodeActAgent/<run_name>/
"""

import asyncio
import copy
import errno
import json
import os
import tempfile
import time
from typing import Any

import pandas as pd
import toml
from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader

import openhands.agenthub
from evaluation.benchmarks.swe_bench.resource.mapping import (
    get_instance_resource_factor,
)
from evaluation.benchmarks.swe_bench_optimized.swe_smith.dataset_utils import (
    build_eval_script,
    get_swesmith_docker_image,
    get_swesmith_workspace_dir_name,
    load_swesmith_dataset,
)
from evaluation.utils.shared import (
    EvalException,
    EvalMetadata,
    EvalOutput,
    assert_and_raise,
    check_maximum_retries_exceeded,
    codeact_user_response,
    get_default_sandbox_config_for_eval,
    get_metrics,
    get_openhands_config_for_eval,
    is_fatal_evaluation_error,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
    update_llm_config_for_completions_logging,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    AgentConfig,
    OpenHandsConfig,
    get_agent_config_arg,
    get_evaluation_parser,
    get_llm_config_arg,
    get_llms_for_routing_config,
    get_model_routing_config_arg,
)
from openhands.core.config.condenser_config import NoOpCondenserConfig
from openhands.core.config.utils import get_condenser_config_arg
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.critic import AgentFinishedCritic
from openhands.events.action import CmdRunAction, FileReadAction, MessageAction
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    FileReadObservation,
)
from openhands.events.serialization.event import event_from_dict, event_to_dict
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync
from openhands.utils.shutdown_listener import sleep_if_should_continue


FAKE_RESPONSES = {
    'CodeActAgent': codeact_user_response,
}
AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = FAKE_RESPONSES

USE_HINT_TEXT = os.environ.get('USE_HINT_TEXT', 'false').lower() == 'true'
RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'

DEFAULT_DOCKER_IMAGE_PREFIX = os.environ.get(
    'EVAL_DOCKER_IMAGE_PREFIX', 'docker.io/xingyaoww/'
)


def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    """Create OpenHands config for a SWE-smith instance."""
    # SWE-smith has image_name directly in the dataset
    base_container_image = instance['image_name']
    logger.info(f'Using SWE-smith container image: {base_container_image}')

    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = base_container_image
    sandbox_config.enable_auto_lint = True
    sandbox_config.use_host_network = False
    sandbox_config.platform = 'linux/amd64'

    # Resource factor
    sandbox_config.remote_runtime_resource_factor = int(
        os.environ.get('DEFAULT_RUNTIME_RESOURCE_FACTOR', 2)
    )

    config = get_openhands_config_for_eval(
        runtime=os.environ.get('RUNTIME', 'docker'),
        sandbox_config=sandbox_config,
    )

    agent_config = get_agent_config_arg(metadata.agent_class, config.get_agent_configs())
    if agent_config and not isinstance(agent_config.condenser, NoOpCondenserConfig):
        condenser_name = os.environ.get('EVAL_CONDENSER')
        if condenser_name:
            condenser_config = get_condenser_config_arg(condenser_name)
            if condenser_config:
                agent_config.condenser = condenser_config

    config.set_llm_config(metadata.llm_config)
    config.set_agent_config(agent_config or AgentConfig())

    return config


def get_instruction(instance: pd.Series, metadata: EvalMetadata) -> MessageAction:
    """Generate instruction for the agent from a SWE-smith instance."""
    workspace_dir_name = get_swesmith_workspace_dir_name(instance)

    # Template selection - use the same templates as SWE-bench
    template_name = metadata.instruction_template_name
    if not template_name:
        llm_model = metadata.llm_config.model
        if 'gpt-4.1' in llm_model:
            template_name = 'swe_gpt4.j2'
        else:
            template_name = 'swe_default.j2'

    # Load template from the parent swe_bench_optimized prompts directory
    prompts_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'prompts',
    )
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    context = {
        'instance': instance,
        'workspace_dir_name': workspace_dir_name,
        'metadata': metadata,
    }

    instruction = template.render(context)

    if RUN_WITH_BROWSING:
        instruction += '\n<IMPORTANT!>\nYou SHOULD NEVER attempt to browse the web.\n</IMPORTANT!>\n'

    return MessageAction(content=instruction)


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,
    metadata: EvalMetadata,
) -> None:
    """Initialize the runtime for a SWE-smith instance.

    SWE-smith containers differ from SWE-bench:
    - The repo is at /testbed
    - Need to fetch and checkout the instance branch from the mirror repo
    - Then checkout HEAD~1 to restore test files
    - The conda env is 'testbed'
    """
    workspace_dir_name = get_swesmith_workspace_dir_name(instance)
    instance_id = instance['instance_id']

    # Set up environment
    action = CmdRunAction(
        command=(
            f'export SWE_INSTANCE_ID={instance_id} && '
            'export PIP_CACHE_DIR=~/.cache/pip && '
            'export PAGER=cat && '
            'export GIT_PAGER=cat && '
            'git config --global core.pager "" && '
            'git config --global diff.binary false'
        )
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to set up environment: {str(obs)}')

    # SWE-smith: The repo is at /testbed (Docker WORKDIR)
    # We need to symlink it to /workspace/testbed if not already done
    action = CmdRunAction(
        command=(
            'if [ ! -d /workspace/testbed ]; then '
            '  ln -sf /testbed /workspace/testbed; '
            'fi && '
            'cd /testbed'
        )
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to set up workspace: {str(obs)}')

    # Fetch the instance branch from the mirror repo
    # The repo field has the GitHub mirror name (e.g., "swesmith/oauthlib__oauthlib.1fd52536")
    repo_mirror = instance.get('repo', '')
    # Instance ID is the branch name in the mirror
    branch_name = instance_id

    action = CmdRunAction(
        command=(
            f'cd /testbed && '
            f'git fetch origin {branch_name} 2>/dev/null || '
            f'git fetch origin 2>/dev/null || true'
        )
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    # fetch may fail if no remote is configured, that's OK

    # Try to checkout the instance branch
    # If the branch exists, checkout and go to HEAD~1 (which restores test files)
    action = CmdRunAction(
        command=(
            f'cd /testbed && '
            f'git checkout {branch_name} 2>/dev/null && '
            f'git checkout HEAD~1 2>/dev/null && '
            f'echo "CHECKOUT_SUCCESS" || echo "CHECKOUT_SKIPPED"'
        )
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if 'CHECKOUT_SKIPPED' in obs.content:
        # Branch doesn't exist locally — the image may already be at the right state
        # (some SWE-smith images are built with the bug already applied)
        logger.warning(
            f'Could not checkout branch {branch_name}. '
            'Assuming image is already at the correct state.'
        )

    # Remove git remotes to prevent the agent from accessing them
    action = CmdRunAction(
        command='cd /testbed && for remote_name in $(git remote); do git remote remove "${remote_name}"; done'
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    # Clean up git history to prevent the agent from seeing the fix
    action = CmdRunAction(
        command=(
            'cd /testbed && '
            'for branch in $(git for-each-ref --format="%(refname:short)" refs/heads/ | grep -v "^$(git rev-parse --abbrev-ref HEAD)$"); do '
            'git branch -D "$branch" 2>/dev/null; done && '
            'for tag in $(git for-each-ref --format="%(refname:short)" refs/tags/); do '
            'git tag -d "$tag" 2>/dev/null; done && '
            'git reflog expire --expire=now --all 2>/dev/null && '
            'git gc --prune=now 2>/dev/null || true'
        )
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    # Activate conda environment and do pip install -e .
    action = CmdRunAction(
        command=(
            'cd /testbed && '
            'source /opt/miniconda3/bin/activate && '
            'conda activate testbed && '
            'which python && '
            'pip install -e . --no-deps -q 2>/dev/null || true'
        )
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    # Non-fatal if pip install fails (some repos don't support it)

    # Redirect editable install paths from /testbed to /workspace/testbed
    # is not needed since they are the same via symlink

    logger.info(f'Runtime initialized for SWE-smith instance {instance_id}')


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,
) -> dict[str, Any]:
    """Extract git patch from the runtime after agent has run."""
    workspace_dir_name = get_swesmith_workspace_dir_name(instance)

    action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)
    if obs.exit_code != 0:
        # Try directly
        action = CmdRunAction(command='cd /testbed')
        action.set_hard_timeout(600)
        obs = runtime.run_action(action)

    # Remove any .git directories in subdirectories to avoid confusing git
    action = CmdRunAction(
        command=(
            'cd /testbed && '
            'find . -mindepth 2 -name .git -type d -exec rm -rf {} + 2>/dev/null || true'
        )
    )
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)

    # Stage all changes
    action = CmdRunAction(command='cd /testbed && git add -A')
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)

    # Remove binary files from staging
    action = CmdRunAction(
        command=(
            'cd /testbed && '
            r"git diff --cached --name-only --diff-filter=d | xargs -I{} sh -c "
            r"'file \"{}\" | grep -qv text && git reset HEAD \"{}\" 2>/dev/null' || true"
        )
    )
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)

    # Generate patch
    action = CmdRunAction(
        command='cd /testbed && git diff --no-color --cached > /tmp/patch.diff 2>&1'
    )
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)

    # Read the patch
    action = FileReadAction(path='/tmp/patch.diff')
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)

    git_patch = ''
    if isinstance(obs, FileReadObservation):
        git_patch = obs.content
    elif isinstance(obs, CmdOutputObservation):
        git_patch = obs.content
    else:
        # Fallback: cat the patch
        action = CmdRunAction(command='cat /tmp/patch.diff')
        action.set_hard_timeout(600)
        obs = runtime.run_action(action)
        if isinstance(obs, CmdOutputObservation):
            git_patch = obs.content

    # Clean up binary diffs if any
    from evaluation.benchmarks.swe_bench.binary_patch_utils import remove_binary_diffs
    git_patch = remove_binary_diffs(git_patch)

    return {'git_patch': git_patch}


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
    runtime_failure_count: int = 0,
) -> EvalOutput:
    """Process a single SWE-smith instance through the agent."""
    instance_id = instance.instance_id

    config = get_config(instance, metadata)

    # Setup logger
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        os.makedirs(log_dir, exist_ok=True)
        reset_logger_for_multiprocessing(logger, instance_id, log_dir)

    # Increase resource_factor for retries
    if runtime_failure_count > 0:
        config.sandbox.remote_runtime_resource_factor = min(
            config.sandbox.remote_runtime_resource_factor * (2**runtime_failure_count),
            8,
        )
        logger.warning(
            f'Retry {runtime_failure_count + 1} for {instance_id}, '
            f'resource factor: {config.sandbox.remote_runtime_resource_factor}'
        )

    metadata = copy.deepcopy(metadata)
    metadata.details['runtime_failure_count'] = runtime_failure_count

    # Set up logging for LLM completions
    if metadata.llm_config.log_completions:
        metadata.llm_config = update_llm_config_for_completions_logging(
            metadata.llm_config,
            instance_id,
            metadata.eval_output_dir,
        )

    try:
        runtime = create_runtime(config)
        call_async_from_sync(runtime.connect)

        initialize_runtime(runtime, instance, metadata)

        instruction = get_instruction(instance, metadata)

        # Determine fake user response function
        fake_user_response_fn = AGENT_CLS_TO_FAKE_USER_RESPONSE_FN.get(
            metadata.agent_class, codeact_user_response
        )

        state: State | None = asyncio.run(
            run_controller(
                config=config,
                initial_user_action=instruction,
                runtime=runtime,
                fake_user_response_fn=fake_user_response_fn,
            )
        )

        # Check for fatal errors
        if is_fatal_evaluation_error(state):
            raise EvalException(
                instance_id,
                'Fatal evaluation error',
                logger,
            )

        # Extract the patch
        test_result = complete_runtime(runtime, instance)

        # Get metrics
        metrics = get_metrics(state) if state else {}

        # Serialize history
        history = []
        if state:
            history = [event_to_dict(e) for e in state.history]

        return EvalOutput(
            instance_id=instance_id,
            instance=instance.to_dict() if hasattr(instance, 'to_dict') else dict(instance),
            instruction=instruction.content if instruction else '',
            metadata=metadata,
            history=history,
            metrics=metrics,
            test_result=test_result,
            error=None,
        )

    except Exception as e:
        logger.error(f'Error processing {instance_id}: {e}')
        if check_maximum_retries_exceeded(runtime_failure_count):
            return EvalOutput(
                instance_id=instance_id,
                instance=instance.to_dict() if hasattr(instance, 'to_dict') else dict(instance),
                instruction='',
                metadata=metadata,
                history=[],
                metrics={},
                test_result={'git_patch': ''},
                error=str(e),
            )
        raise

    finally:
        try:
            runtime.close()
        except Exception:
            pass


if __name__ == '__main__':
    parser = get_evaluation_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='SWE-bench/SWE-smith',
        help='SWE-smith dataset name on HuggingFace',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='Dataset split',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='swe',
        choices=['swe'],
        help='Evaluation mode',
    )
    parser.add_argument(
        '--n-runs',
        type=int,
        default=1,
        help='Number of runs per instance',
    )
    parser.add_argument(
        '--filter-real-only',
        action='store_true',
        default=True,
        help='Filter to real PR-mirror instances only (default: True)',
    )
    parser.add_argument(
        '--no-filter-real-only',
        action='store_false',
        dest='filter_real_only',
        help='Include all SWE-smith instances (not just real PR ones)',
    )

    args, _ = parser.parse_known_args()

    # Load dataset
    swe_smith_tests = load_swesmith_dataset(
        dataset_name=args.dataset,
        split=args.split,
        filter_real_only=args.filter_real_only,
    )
    logger.info(f'Loaded {len(swe_smith_tests)} SWE-smith instances for evaluation')

    # Set up LLM config
    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config, args.config_file)
        llm_config.log_completions = True
        llm_config.modify_params = False

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    condenser_name = os.environ.get('EVAL_CONDENSER')
    condenser_config = None
    if condenser_name:
        condenser_config = get_condenser_config_arg(condenser_name, args.config_file)

    metadata = make_metadata(
        llm_config,
        args.dataset,
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
        details={
            'mode': args.mode,
            'dataset_type': 'SWE-smith',
        },
    )

    output_file, instance_ids_done, _ = prepare_dataset(
        swe_smith_tests, metadata.eval_output_dir, args.eval_n_limit
    )

    # Filter already done instances
    swe_smith_tests = swe_smith_tests[
        ~swe_smith_tests['instance_id'].isin(instance_ids_done)
    ]
    logger.info(
        f'{len(swe_smith_tests)} instances remaining after filtering already done'
    )

    run_evaluation(
        swe_smith_tests,
        metadata=metadata,
        output_file=output_file,
        num_workers=args.eval_num_workers,
        process_instance_func=process_instance,
    )
