"""
Replay and Refine Script for SWE-bench Trajectories

This script allows you to:
1. Load an existing trajectory from a previous inference run
2. Identify a specific step where the agent might have gone wrong
3. Replay the trajectory up to that step to restore the environment state
4. Continue inference from that point with a refined/suggested action
5. Generate a new refined trajectory

Usage:
    python replay_and_refine.py \
        --trajectory-path /path/to/original/trajectory.json \
        --refinement-input /path/to/refinement_input.json \
        --output-dir /path/to/output \
        --model-config llm

    Refinement input JSON format:
    {
        "instance_id": "django__django-12345",
        "target_step_id": 15,
        "reason": "Wrong file was edited",
        "bad_consequence": "Test failure due to incorrect modification",
        "suggested_action": {
            "action": "edit",
            "args": {
                "path": "/workspace/correct_file.py",
                "content": "..."
            },
            "thought": "Edit the correct file instead"
        }
    }
"""

import argparse
import asyncio
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import openhands.agenthub
from evaluation.benchmarks.swe_bench.run_infer import (
    complete_runtime,
    get_config,
    get_instruction,
    initialize_runtime,
)
from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    get_metrics,
    make_metadata,
    reset_logger_for_multiprocessing,
)
from openhands.controller.state.state import State
from openhands.core.config import OpenHandsConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.loop import run_agent_until_done
from openhands.core.main import create_runtime
from openhands.core.schema.agent import AgentState
from openhands.core.setup import create_agent, create_controller, create_memory
from openhands.events.action import Action, MessageAction
from openhands.events.event import Event, EventSource
from openhands.events.serialization.event import event_from_dict, event_to_dict
from openhands.runtime.base import Runtime
from openhands.server.services.conversation_stats import ConversationStats
from openhands.utils.async_utils import call_async_from_sync
from openhands.utils.utils import create_registry_and_conversation_stats


class RefinementInput:
    """Data class for refinement input"""

    def __init__(
        self,
        instance_id: str,
        target_step_id: int,
        reason: str,
        bad_consequence: str,
        suggested_action: dict[str, Any],
    ):
        self.instance_id = instance_id
        self.target_step_id = target_step_id
        self.reason = reason
        self.bad_consequence = bad_consequence
        self.suggested_action = suggested_action

    @classmethod
    def from_dict(cls, data: dict) -> 'RefinementInput':
        """Create RefinementInput from dictionary"""
        return cls(
            instance_id=data['instance_id'],
            target_step_id=data['target_step_id'],
            reason=data.get('reason', ''),
            bad_consequence=data.get('bad_consequence', ''),
            suggested_action=data['suggested_action'],
        )

    @classmethod
    def from_json_file(cls, path: str) -> 'RefinementInput':
        """Load RefinementInput from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def load_trajectory(trajectory_path: str) -> tuple[list[Event], dict[str, Any]]:
    """
    Load trajectory from JSON file and convert to Event objects.

    Args:
        trajectory_path: Path to the trajectory JSON file

    Returns:
        Tuple of (events list, metadata dict)
    """
    logger.info(f'Loading trajectory from: {trajectory_path}')

    with open(trajectory_path, 'r') as f:
        trajectory_data = json.load(f)

    # Extract events and metadata
    if isinstance(trajectory_data, dict):
        events_data = trajectory_data.get('history', [])
        metadata = {
            k: v for k, v in trajectory_data.items() if k not in ['history', 'test_result']
        }
        instance_data = trajectory_data.get('instance', {})
    else:
        # Assume it's a list of events
        events_data = trajectory_data
        metadata = {}
        instance_data = {}

    # Convert to Event objects
    events = []
    for event_dict in events_data:
        try:
            event = event_from_dict(event_dict)
            events.append(event)
        except Exception as e:
            logger.warning(f'Failed to convert event: {e}')
            continue

    logger.info(f'Loaded {len(events)} events from trajectory')
    return events, {'metadata': metadata, 'instance': instance_data}


def extract_partial_trajectory(
    events: list[Event], target_step_id: int
) -> tuple[list[Event], Event | None]:
    """
    Extract events up to (but not including) the target step.

    Args:
        events: Full list of events
        target_step_id: ID of the step to stop before

    Returns:
        Tuple of (partial events list, target event that was removed)
    """
    logger.info(f'Extracting trajectory up to step {target_step_id}')

    partial_events = []
    target_event = None

    for event in events:
        event_id = getattr(event, 'id', None)
        if event_id is not None and event_id >= target_step_id:
            if event_id == target_step_id:
                target_event = event
                logger.info(f'Found target event at id={event_id}: {type(event).__name__}')
            break
        partial_events.append(event)

    logger.info(f'Extracted {len(partial_events)} events before step {target_step_id}')
    return partial_events, target_event


def create_suggested_action_from_dict(action_dict: dict[str, Any]) -> Action:
    """
    Create an Action object from the suggested action dictionary.

    Args:
        action_dict: Dictionary with 'action', 'args', and optional 'thought'

    Returns:
        Action object
    """
    action_type = action_dict.get('action', 'message')
    action_args = action_dict.get('args', {})
    thought = action_dict.get('thought', '')

    # Import action classes dynamically
    from openhands.events.action import (
        AgentFinishAction,
        CmdRunAction,
        FileEditAction,
        FileReadAction,
        MessageAction,
    )

    # Map action type strings to classes
    action_map = {
        'message': MessageAction,
        'cmd_run': CmdRunAction,
        'run': CmdRunAction,
        'edit': FileEditAction,
        'file_edit': FileEditAction,
        'read': FileReadAction,
        'file_read': FileReadAction,
        'finish': AgentFinishAction,
    }

    action_cls = action_map.get(action_type.lower())
    if action_cls is None:
        logger.warning(
            f'Unknown action type: {action_type}, defaulting to MessageAction'
        )
        action_cls = MessageAction
        action_args = {'content': f"Suggested action: {action_dict}"}

    # Add thought if provided and action supports it
    if thought and hasattr(action_cls, 'thought'):
        action_args['thought'] = thought

    try:
        action = action_cls(**action_args)
        return action
    except Exception as e:
        logger.error(f'Failed to create action from dict: {e}')
        # Fallback to a message action
        return MessageAction(content=f"Suggested action: {action_dict}")


async def replay_and_refine(
    instance: pd.Series,
    original_events: list[Event],
    refinement_input: RefinementInput,
    metadata: EvalMetadata,
    output_dir: str,
) -> EvalOutput:
    """
    Main function to replay trajectory up to a point and continue with refined action.

    This function mimics the structure and flow of process_instance() in run_infer.py
    to ensure compatibility with the evaluation pipeline.

    Args:
        instance: SWE-bench instance data
        original_events: Full list of events from original trajectory
        refinement_input: Refinement instructions
        metadata: Evaluation metadata
        output_dir: Directory to save output

    Returns:
        EvalOutput with refined trajectory
    """
    logger.info('=' * 80)
    logger.info(f'Starting replay and refinement for {instance.instance_id}')
    logger.info(f'Target step: {refinement_input.target_step_id}')
    logger.info(f'Reason: {refinement_input.reason}')
    logger.info(f'Bad consequence: {refinement_input.bad_consequence}')
    logger.info('=' * 80)

    # Extract partial trajectory
    partial_events, target_event = extract_partial_trajectory(
        original_events, refinement_input.target_step_id
    )

    if not partial_events:
        raise ValueError(
            f'No events found before step {refinement_input.target_step_id}'
        )

    # Get configuration - EXACTLY like run_infer.py does
    config = get_config(instance, metadata)

    # Set up replay mode with partial trajectory
    # The first event should be a MessageAction (initial instruction)
    initial_message = None
    replay_events = []

    for event in partial_events:
        if isinstance(event, MessageAction) and initial_message is None:
            initial_message = event
        else:
            # Clear event IDs for replay
            event._id = None  # type: ignore
            replay_events.append(event)

    if initial_message is None:
        # If no MessageAction found, get the instruction from instance
        # EXACTLY like run_infer.py does
        initial_message = get_instruction(instance, metadata)
        logger.info('Using original instruction as initial message')
    else:
        # Clear the ID from the initial message so it can be added to event stream
        initial_message._id = None  # type: ignore
        logger.info('Using MessageAction from trajectory as initial message')

    # Create runtime and initialize - EXACTLY like run_infer.py
    runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)  # Use call_async_from_sync like run_infer

    try:
        # Initialize runtime - REUSING the exact function from run_infer.py
        initialize_runtime(runtime, instance, metadata)

        logger.info(
            f'Replaying {len(replay_events)} events to restore state up to step {refinement_input.target_step_id}'
        )

        # Create llm_registry and conversation_stats - EXACTLY like run_controller does
        llm_registry, conversation_stats, _ = create_registry_and_conversation_stats(
            config,
            runtime.sid,
            None,
        )

        # Create agent - REUSING the exact pattern from run_controller
        agent = create_agent(config, llm_registry)

        # Create memory - EXACTLY like run_controller does
        memory = create_memory(
            runtime=runtime,
            event_stream=runtime.event_stream,
            sid=runtime.sid,
            selected_repository=config.sandbox.selected_repo,
            repo_directory=None,
            conversation_instructions=None,
            working_dir=str(runtime.workspace_root),
        )

        # Create controller with replay_events
        controller, initial_state = create_controller(
            agent=agent,
            runtime=runtime,
            config=config,
            conversation_stats=conversation_stats,
            headless_mode=True,
            replay_events=replay_events,
        )

        logger.info('Controller created with replay events')

        # Add the initial message to event stream
        runtime.event_stream.add_event(initial_message, EventSource.USER)

        # Define end states for agent execution
        end_states = [
            AgentState.FINISHED,
            AgentState.REJECTED,
            AgentState.ERROR,
            AgentState.PAUSED,
            AgentState.STOPPED,
        ]

        # Now inject the refinement guidance as a user message BEFORE starting replay
        # This way the agent will see it when replay completes and continue naturally
        logger.info('=' * 80)
        logger.info('Injecting refinement guidance...')
        logger.info(f"Target step: {refinement_input.target_step_id}")
        logger.info(f"Reason: {refinement_input.reason}")
        logger.info('=' * 80)

        # Format the refinement as a clear instruction to the agent
        action_description = json.dumps(refinement_input.suggested_action, indent=2)
        refinement_content = f"""[Refinement Guidance]

Based on trajectory analysis, at step {refinement_input.target_step_id}, there was an issue:

**Reason**: {refinement_input.reason}
**Consequence if not corrected**: {refinement_input.bad_consequence}

**Suggested correction**:
```json
{action_description}
```

Please take this guidance into account and continue working to complete the task correctly."""

        refinement_message = MessageAction(
            content=refinement_content,
            wait_for_response=False,
        )

        logger.info(f'Adding refinement guidance:\n{refinement_content}')

        # Add the refinement message BEFORE replay, so it's ready when replay completes
        runtime.event_stream.add_event(refinement_message, EventSource.USER)

        # Run agent - this will replay events first, then continue with refinement
        await run_agent_until_done(
            controller=controller,
            runtime=runtime,
            memory=memory,
            end_states=end_states,
        )

        # Get final state from controller
        state = controller.get_state()
        if state is None:
            raise ValueError('State should not be None after execution')

        logger.info(f'Execution complete. Final agent state: {state.agent_state}')
        logger.info(f'Final history length: {len(state.history)}')

        # Get git patch - EXACTLY like run_infer.py does
        # Import DATASET_TYPE from run_infer to check dataset type
        from evaluation.benchmarks.swe_bench.run_infer import DATASET_TYPE

        if DATASET_TYPE == 'SWE-bench-Live':
            from evaluation.benchmarks.swe_bench.live_utils import (
                complete_runtime as complete_runtime_fn,
            )
        else:
            complete_runtime_fn = complete_runtime

        return_val = complete_runtime_fn(runtime, instance)
        git_patch = return_val['git_patch']
        logger.info(
            f'Got git diff for refined instance {instance.instance_id}:\n--------\n{git_patch}\n--------'
        )

    finally:
        # Always close runtime - EXACTLY like run_infer.py
        runtime.close()

    # ======= Prepare output - EXACTLY like run_infer.py =======
    test_result = {
        'git_patch': git_patch,
    }

    if state is None:
        raise ValueError('State should not be None.')

    # NOTE: this is NO LONGER the event stream, but an agent history that includes delegate agent's events
    histories = [event_to_dict(event) for event in state.history]
    metrics = get_metrics(state)

    # Save metadata about the refinement
    refinement_metadata = {
        'original_trajectory_step_count': len(original_events),
        'replay_step_count': len(partial_events),
        'target_step_id': refinement_input.target_step_id,
        'refinement_reason': refinement_input.reason,
        'bad_consequence': refinement_input.bad_consequence,
        'suggested_action': refinement_input.suggested_action,
        'target_event_type': type(target_event).__name__ if target_event else None,
    }

    metadata_dict = metadata.model_dump()
    metadata_dict['refinement'] = refinement_metadata

    # Create output - EXACTLY matching run_infer.py format
    instruction = initial_message.content
    if hasattr(initial_message, 'image_urls') and initial_message.image_urls:
        instruction += (
            '\n\n<image_urls>'
            + '\n'.join(initial_message.image_urls)
            + '</image_urls>'
        )

    output = EvalOutput(
        instance_id=instance.instance_id,  # Keep original ID for evaluation
        instruction=instruction,
        instance=instance.to_dict(),  # SWE Bench specific
        test_result=test_result,
        metadata=metadata_dict,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
    )

    return output


def process_refinement(
    trajectory_path: str,
    refinement_input: RefinementInput,
    output_dir: str,
    model_config: str,
    agent_class: str = 'CodeActAgent',
    max_iterations: int = 50,
    eval_note: str = 'replay_refine',
    dataset_name: str = 'SWE-bench',
) -> None:
    """
    Process a single refinement request.

    This function mimics the structure of process_instance() in run_infer.py
    to ensure consistency with the original evaluation pipeline.

    Args:
        trajectory_path: Path to original trajectory JSON
        refinement_input: Refinement instructions
        output_dir: Directory to save output
        model_config: LLM config name
        agent_class: Agent class name
        max_iterations: Max iterations for continuation
        eval_note: Note for evaluation output
        dataset_name: Dataset name (default: SWE-bench)
    """
    # Load the original trajectory
    original_events, trajectory_metadata = load_trajectory(trajectory_path)

    # Extract instance data
    instance_data = trajectory_metadata.get('instance', {})
    if not instance_data:
        raise ValueError('No instance data found in trajectory')

    # Create pandas Series from instance data (same as run_infer.py)
    instance = pd.Series(instance_data)

    # Verify instance ID matches
    if instance.get('instance_id') != refinement_input.instance_id:
        logger.warning(
            f"Instance ID mismatch: trajectory has {instance.get('instance_id')}, "
            f"refinement input has {refinement_input.instance_id}"
        )

    # Create metadata - MATCHING run_infer.py structure
    from openhands.core.config import get_llm_config_arg

    llm_config = get_llm_config_arg(model_config)
    llm_config.log_completions = True  # Enable logging like in run_infer
    llm_config.modify_params = False   # Disable param modification for consistency

    metadata = make_metadata(
        llm_config=llm_config,
        dataset_name=dataset_name,
        agent_class=agent_class,
        max_iterations=max_iterations,
        eval_output_dir=output_dir,
        eval_note=eval_note,
    )

    # Add refinement info to metadata details
    metadata.details = metadata.details or {}
    metadata.details['mode'] = 'swe'  # Default mode, matching run_infer.py
    metadata.details['is_refinement'] = True
    metadata.details['original_instance_id'] = instance.instance_id
    metadata.details['target_step_id'] = refinement_input.target_step_id
    metadata.details['refinement_reason'] = refinement_input.reason

    # Set up logging - MATCHING run_infer.py
    log_dir = os.path.join(output_dir, 'infer_logs')
    os.makedirs(log_dir, exist_ok=True)
    reset_logger_for_multiprocessing(
        logger, f"{instance.instance_id}_refined", log_dir
    )

    # Run replay and refinement
    output = asyncio.run(
        replay_and_refine(
            instance=instance,
            original_events=original_events,
            refinement_input=refinement_input,
            metadata=metadata,
            output_dir=output_dir,
        )
    )

    # Save output to JSONL format (matching run_infer.py output format)
    output_file = os.path.join(output_dir, 'output.jsonl')
    os.makedirs(output_dir, exist_ok=True)

    # Append to output.jsonl (same format as run_infer.py)
    with open(output_file, 'a') as f:
        f.write(output.model_dump_json() + '\n')

    logger.info(f'Saved refined trajectory to: {output_file}')


def batch_process_refinements(
    trajectory_dir: str,
    refinement_inputs_file: str,
    output_dir: str,
    model_config: str,
    agent_class: str = 'CodeActAgent',
    max_iterations: int = 50,
    eval_note: str = 'replay_refine_batch',
    dataset_name: str = 'SWE-bench',
) -> None:
    """
    Process multiple refinement requests from a batch file.

    Output format matches run_infer.py - all results in output.jsonl

    Args:
        trajectory_dir: Directory containing original trajectory JSON files
        refinement_inputs_file: JSON file with list of refinement inputs
        output_dir: Directory to save outputs
        model_config: LLM config name
        agent_class: Agent class name
        max_iterations: Max iterations for continuation
        eval_note: Note for evaluation output
        dataset_name: Dataset name (default: SWE-bench)
    """
    logger.info(f'Loading refinement inputs from: {refinement_inputs_file}')

    with open(refinement_inputs_file, 'r') as f:
        refinement_data = json.load(f)

    # Support both single dict and list of dicts
    if isinstance(refinement_data, dict):
        refinement_data = [refinement_data]

    logger.info(f'Processing {len(refinement_data)} refinement requests')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'output.jsonl')
    logger.info(f'Output will be saved to: {output_file}')

    for idx, ref_dict in enumerate(refinement_data):
        logger.info('=' * 80)
        logger.info(f'Processing refinement {idx + 1}/{len(refinement_data)}')
        logger.info('=' * 80)

        try:
            refinement_input = RefinementInput.from_dict(ref_dict)

            # Find trajectory file
            trajectory_file = os.path.join(
                trajectory_dir, f"{refinement_input.instance_id}.json"
            )

            if not os.path.exists(trajectory_file):
                logger.error(f'Trajectory file not found: {trajectory_file}')
                continue

            # Process refinement
            process_refinement(
                trajectory_path=trajectory_file,
                refinement_input=refinement_input,
                output_dir=output_dir,
                model_config=model_config,
                agent_class=agent_class,
                max_iterations=max_iterations,
                eval_note=eval_note,
                dataset_name=dataset_name,
            )

        except Exception as e:
            logger.error(f'Failed to process refinement {idx + 1}: {e}', exc_info=True)
            continue

    logger.info('=' * 80)
    logger.info(f'Batch processing complete! Results saved to: {output_file}')
    logger.info('=' * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Replay and refine SWE-bench trajectories'
    )

    parser.add_argument(
        '--trajectory-path',
        type=str,
        help='Path to original trajectory JSON file (for single refinement)',
    )
    parser.add_argument(
        '--trajectory-dir',
        type=str,
        help='Directory containing trajectory JSON files (for batch processing)',
    )
    parser.add_argument(
        '--refinement-input',
        type=str,
        required=True,
        help='Path to refinement input JSON file',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save refined trajectories',
    )
    parser.add_argument(
        '--model-config',
        type=str,
        default='llm',
        help='LLM config name from config.toml (default: llm)',
    )
    parser.add_argument(
        '--agent-class',
        type=str,
        default='CodeActAgent',
        help='Agent class name (default: CodeActAgent)',
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=50,
        help='Maximum iterations for continuation (default: 50)',
    )
    parser.add_argument(
        '--eval-note',
        type=str,
        default='replay_refine',
        help='Evaluation note (default: replay_refine)',
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process multiple refinements from batch file',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='SWE-bench',
        help='Dataset name (default: SWE-bench)',
    )

    args = parser.parse_args()

    # Validate arguments
    if args.batch:
        if not args.trajectory_dir:
            parser.error('--trajectory-dir is required for batch processing')
        batch_process_refinements(
            trajectory_dir=args.trajectory_dir,
            refinement_inputs_file=args.refinement_input,
            output_dir=args.output_dir,
            model_config=args.model_config,
            agent_class=args.agent_class,
            max_iterations=args.max_iterations,
            eval_note=args.eval_note,
            dataset_name=args.dataset,
        )
    else:
        if not args.trajectory_path:
            parser.error('--trajectory-path is required for single refinement')
        refinement_input = RefinementInput.from_json_file(args.refinement_input)
        process_refinement(
            trajectory_path=args.trajectory_path,
            refinement_input=refinement_input,
            output_dir=args.output_dir,
            model_config=args.model_config,
            agent_class=args.agent_class,
            max_iterations=args.max_iterations,
            eval_note=args.eval_note,
            dataset_name=args.dataset,
        )


if __name__ == '__main__':
    main()
