"""GuidedCodeActAgent — a CodeActAgent with a Blinded Critic validation loop.

The agent receives the standard issue description PLUS a sealed
``<REFERENCE_INFORMATION>`` block containing the golden patch and golden test
(see ``prompts/swe_guided.j2``).  At every step, before accepting the agent's
action, a :class:`BlindedCritic` that has *never* seen the golden data reviews
the response.  If the critic rejects it, the critique is fed back to the agent
so it can try again, up to ``BLINDED_CRITIC_MAX_RETRIES`` times.

This enforces that the full trajectory is reachable from accumulated evidence
rather than directly copying the golden answer.

Validation log persistence
--------------------------
Because ``run_controller`` constructs and owns the agent internally, callers
cannot directly access the agent object after the run.  Each
:class:`GuidedCodeActAgent` therefore writes every validation entry to a
per-process JSONL file::

    /tmp/blinded_critic_<PID>.jsonl

Callers can read/clear this file via the helpers
:func:`get_validation_log_path`, :func:`clear_validation_log`, and
:func:`read_and_clear_validation_log`.
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Per-process validation log helpers
# ---------------------------------------------------------------------------

def get_validation_log_path() -> str:
    """Return the path to the per-process JSONL validation log file."""
    return f'/tmp/blinded_critic_{os.getpid()}.jsonl'


def clear_validation_log() -> None:
    """Delete the current process's validation log file if it exists."""
    path = get_validation_log_path()
    if os.path.exists(path):
        os.remove(path)


def read_and_clear_validation_log() -> list[dict]:
    """Read all entries from the per-process log file, then delete it.

    Returns an empty list if the file does not exist.
    """
    path = get_validation_log_path()
    entries: list[dict] = []
    if not os.path.exists(path):
        return entries
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception:
        pass
    try:
        os.remove(path)
    except Exception:
        pass
    return entries


def _append_validation_entry(entry: dict) -> None:
    """Append a single validation entry to the per-process JSONL log file."""
    path = get_validation_log_path()
    try:
        with open(path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception as exc:
        from openhands.core.logger import openhands_logger as _logger
        _logger.warning(f'[GuidedCodeActAgent] Failed to write validation log: {exc}')


# ---------------------------------------------------------------------------

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.agenthub.guided_codeact_agent.blinded_critic import (
    BlindedCritic,
    ValidationResult,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.events.action import AgentFinishAction, MessageAction
from openhands.events.event import Event
from openhands.events.observation import Observation
from openhands.llm.llm_utils import check_tools
from openhands.memory.condenser.condenser import Condensation, View

if TYPE_CHECKING:
    from openhands.controller.state.state import State
    from openhands.events.action import Action


# Regex to detect and strip the golden reference block from the task instruction
_REFERENCE_BLOCK_RE = re.compile(
    r'<REFERENCE_INFORMATION>.*?</REFERENCE_INFORMATION>',
    re.DOTALL,
)


class GuidedCodeActAgent(CodeActAgent):
    """CodeActAgent extended with an in-loop Blinded Critic validation step.

    On each call to :meth:`step`, after the primary LLM produces a response,
    the :class:`BlindedCritic` judges whether the response is derivable from
    accumulated evidence.  If not, the critic's feedback is injected into the
    message context and the primary LLM is asked to reconsider — up to
    ``BLINDED_CRITIC_MAX_RETRIES`` times.

    The full validation log (one entry per LLM call, including retries) is
    accumulated in :attr:`validation_log` and can be retrieved after the
    trajectory completes.
    """

    VERSION = '1.0'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Lazily initialised the first time step() is called
        self._blinded_critic: BlindedCritic | None = None
        self._critic_initialised: bool = False
        # Accumulated per-step validation results (list of dicts, JSON-serialisable)
        self.validation_log: list[dict] = []

        self._max_retries: int = int(
            os.environ.get('BLINDED_CRITIC_MAX_RETRIES', '3')
        )

    # ------------------------------------------------------------------
    # Core step logic
    # ------------------------------------------------------------------

    def step(self, state: 'State') -> 'Action':
        """Run one step with optional Blinded Critic validation loop."""

        # ---- Early exits identical to CodeActAgent ----
        if self.pending_actions:
            return self.pending_actions.popleft()

        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        # ---- Build condensed history ----
        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events
            case Condensation(action=condensation_action):
                return condensation_action

        logger.debug(
            f'[GuidedCodeActAgent] Processing {len(condensed_history)} events '
            f'(total {len(state.history)})'
        )

        # ---- Lazily initialise the Blinded Critic using the issue text ----
        if not self._critic_initialised:
            self._init_blinded_critic(state)

        # ---- Build messages for primary LLM ----
        initial_user_message = self._get_initial_user_message(state.history)
        messages = self._get_messages(condensed_history, initial_user_message)
        params: dict = {
            'messages': messages,
            'tools': check_tools(self.tools, self.llm.config),
            'extra_body': {
                'metadata': state.to_llm_metadata(
                    model_name=self.llm.config.model, agent_name=self.name
                )
            },
        }

        # ---- Validation loop ----
        history_text = self._render_history_text(condensed_history)
        response = None
        step_index = state.iteration_flag.current_value

        for attempt in range(self._max_retries + 1):
            response = self.llm.completion(**params)
            logger.debug(f'[GuidedCodeActAgent] Response (attempt {attempt}): {response}')

            if self._blinded_critic is None:
                # No critic configured; proceed immediately
                break

            response_text = self._extract_response_text(response)
            validation: ValidationResult = self._blinded_critic.validate(
                step_index=step_index,
                history_text=history_text,
                agent_response_text=response_text,
                attempt=attempt,
            )

            # Record in memory and persist to per-process log file
            entry = {
                'step_index': step_index,
                'attempt': attempt,
                **validation.to_dict(),
            }
            self.validation_log.append(entry)
            _append_validation_entry(entry)

            if validation.valid:
                logger.debug(
                    f'[GuidedCodeActAgent] Step {step_index} attempt {attempt}: '
                    'VALID — proceeding.'
                )
                break

            if attempt < self._max_retries:
                logger.info(
                    f'[GuidedCodeActAgent] Step {step_index} attempt {attempt}: INVALID — retrying.\n'
                    f'  Reason      : {validation.reason}\n'
                    f'  Unjustified : {validation.unjustified_knowledge}\n'
                    f'  Prereqs     : {validation.prerequisite_conditions}\n'
                    f'  Agent resp  : {response_text[:400]}'
                )
                params['messages'] = self._inject_critique(
                    params['messages'], validation.feedback_message
                )
            else:
                logger.warning(
                    f'[GuidedCodeActAgent] Step {step_index}: max retries ({self._max_retries}) reached — '
                    f'accepting despite validation failure.\n'
                    f'  Reason      : {validation.reason}\n'
                    f'  Agent resp  : {response_text[:400]}'
                )

        # ---- Parse and queue actions ----
        actions = self.response_to_actions(response)
        for action in actions:
            self.pending_actions.append(action)
        return self.pending_actions.popleft()

    def reset(self) -> None:
        super().reset()
        # Keep validation_log across resets; only clear pending actions (done by parent)

    # ------------------------------------------------------------------
    # Blinded Critic initialisation
    # ------------------------------------------------------------------

    def _init_blinded_critic(self, state: 'State') -> None:
        """Initialise the critic using the issue text extracted from the first
        user message, with the golden reference block stripped out."""
        self._critic_initialised = True
        try:
            raw_instruction = self._get_raw_initial_instruction(state.history)
            issue_text = self._strip_reference_block(raw_instruction)
            self._blinded_critic = BlindedCritic.from_env(issue_text=issue_text)
        except Exception as e:
            logger.warning(
                f'[GuidedCodeActAgent] Failed to initialise Blinded Critic: {e}. '
                'Proceeding without validation.'
            )
            self._blinded_critic = None

    def _get_raw_initial_instruction(self, history: list[Event]) -> str:
        """Return the text of the very first user MessageAction."""
        for event in history:
            if isinstance(event, MessageAction) and event.source == 'user':
                return event.content
        return ''

    @staticmethod
    def _strip_reference_block(text: str) -> str:
        """Remove the <REFERENCE_INFORMATION> block so the critic is blind to it."""
        return _REFERENCE_BLOCK_RE.sub('', text).strip()

    # ------------------------------------------------------------------
    # Helpers for rendering history and response as plain text
    # ------------------------------------------------------------------

    @staticmethod
    def _render_history_text(events: list[Event]) -> str:
        """Convert the condensed event history to a plain-text format
        suitable for the Blinded Critic's context window.

        Produces two sections:
        1. A compact "Files & Commands Observed" index so the critic can
           quickly see what the agent has already accessed.
        2. The full step-by-step action/observation log.

        Observation content is included at generous length so the critic
        can verify that specific values appeared in actual tool output.
        The history may be truncated by the condenser, so observations
        are shown with as much context as practical.
        """
        from openhands.events.action import (
            CmdRunAction,
            FileEditAction,
            FileReadAction,
        )
        from openhands.events.action.agent import AgentThinkAction
        from openhands.events.observation import (
            CmdOutputObservation,
            FileEditObservation,
            FileReadObservation,
        )

        # --- Pass 1: build "files/commands observed" index ---
        files_read: list[str] = []
        files_edited: list[str] = []
        commands_run: list[str] = []
        for event in events:
            if isinstance(event, FileReadAction):
                if event.path not in files_read:
                    files_read.append(event.path)
            elif isinstance(event, FileEditAction):
                if event.path not in files_edited:
                    files_edited.append(event.path)
            elif isinstance(event, CmdRunAction):
                commands_run.append(event.command[:120])

        index_lines: list[str] = ['=== SESSION INDEX (what the agent has observed so far) ===']
        if files_read:
            index_lines.append('Files READ this session (agent has seen full content of each):')
            for p in files_read:
                index_lines.append(f'  • {p}')
        if files_edited:
            index_lines.append('Files EDITED this session:')
            for p in files_edited:
                index_lines.append(f'  • {p}')
        if commands_run:
            index_lines.append(f'Commands run this session: {len(commands_run)} total')
        index_lines.append('=== END SESSION INDEX ===')
        index_lines.append('')

        # --- Pass 2: step-by-step log ---
        lines: list[str] = []
        for i, event in enumerate(events):
            if isinstance(event, MessageAction) and event.source == 'user':
                # Skip the initial task instruction (too long; critic already has it)
                continue
            if isinstance(event, MessageAction) and event.source == 'agent':
                lines.append(f'[Event {i}] AGENT MESSAGE: {event.content[:800]}')
            elif isinstance(event, CmdRunAction):
                lines.append(f'[Event {i}] RUN COMMAND: {event.command[:600]}')
            elif isinstance(event, FileReadAction):
                lines.append(f'[Event {i}] READ FILE: {event.path}')
            elif isinstance(event, FileEditAction):
                lines.append(
                    f'[Event {i}] EDIT FILE: {event.path} — '
                    f'{event.content[:400] if hasattr(event, "content") else ""}'
                )
            elif hasattr(event, 'thought') and getattr(event, 'thought', ''):
                lines.append(f'[Event {i}] THINK: {event.thought[:400]}')
            elif isinstance(event, CmdOutputObservation):
                lines.append(
                    f'[Event {i}] OBS (exit={event.exit_code}): '
                    f'{event.content[:2000]}'
                )
            elif isinstance(event, FileReadObservation):
                lines.append(
                    f'[Event {i}] OBS (file read — {len(event.content)} chars total, '
                    f'first 3000 shown): {event.content[:3000]}'
                )
            elif isinstance(event, FileEditObservation):
                lines.append(
                    f'[Event {i}] OBS (file edit): {event.content[:600]}'
                )
            elif isinstance(event, Observation):
                lines.append(
                    f'[Event {i}] OBS: {str(event.content)[:800]}'
                )

        body = '\n'.join(lines) if lines else '(no prior interactions)'
        return '\n'.join(index_lines) + body

    @staticmethod
    def _extract_response_text(response) -> str:
        """Convert a litellm ModelResponse to a human-readable string for validation."""
        if not response or not response.choices:
            return '(empty response)'

        choice = response.choices[0]
        msg = choice.message
        parts: list[str] = []

        # Thought / reasoning content
        content = msg.content
        if content:
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        parts.append(block.get('text', ''))

        # Tool calls (intended actions)
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                fn = tc.function
                parts.append(
                    f'[TOOL CALL] {fn.name}({fn.arguments})'
                )

        return '\n'.join(parts) if parts else '(empty response)'

    @staticmethod
    def _inject_critique(
        messages: list[Message], feedback: str
    ) -> list[Message]:
        """Append the critic's feedback as a user message to the message list."""
        critique_msg = Message(
            role='user',
            content=[
                TextContent(
                    text=(
                        feedback
                        + '\n\nPlease provide a revised response that addresses '
                        'the above concerns.'
                    )
                )
            ],
        )
        return list(messages) + [critique_msg]
