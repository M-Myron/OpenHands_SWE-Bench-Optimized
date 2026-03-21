from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.agenthub.oracle_triad_codeact_agent.oracle_planner import (
    OraclePlanner,
    PlannerDecision,
)
from openhands.agenthub.oracle_triad_codeact_agent.proposal_critic import (
    OracleProposalCritic,
    ProposalValidationResult,
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


_REFERENCE_BLOCK_RE = re.compile(
    r'<REFERENCE_INFORMATION>.*?</REFERENCE_INFORMATION>',
    re.DOTALL,
)

_ISSUE_UNDERSTANDING_BLOCK_RE = re.compile(
    r'<ISSUE_UNDERSTANDING>.*?</ISSUE_UNDERSTANDING>',
    re.DOTALL,
)

_ISSUE_DESCRIPTION_BLOCK_RE = re.compile(
    r'<issue_description>.*?</issue_description>',
    re.DOTALL | re.IGNORECASE,
)


def get_triage_log_path() -> str:
    return f'/tmp/oracle_triad_{os.getpid()}.jsonl'


def clear_triage_log() -> None:
    path = get_triage_log_path()
    if os.path.exists(path):
        os.remove(path)


def read_and_clear_triage_log() -> list[dict]:
    path = get_triage_log_path()
    entries: list[dict] = []
    if not os.path.exists(path):
        return entries

    try:
        with open(path, 'r', encoding='utf-8') as f:
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


def _append_triage_entry(entry: dict) -> None:
    try:
        with open(get_triage_log_path(), 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception as exc:
        logger.warning(f'[OracleTriadCodeActAgent] Failed to write triad log: {exc}')


class OracleTriadCodeActAgent(CodeActAgent):
    """Triad agent with blinded debugger, oracle planner, and blinded critic.

    Flow per step:
    1) Blinded debugger (primary agent LLM) generates N candidate responses.
    2) Oracle planner (oracle-aware LLM) inspects full interaction history and
       candidates, then either selects one candidate or proposes a revised
       response while still being non-leaky and history-grounded.
    3) If planner proposes a revised response, blinded proposal critic validates
       the proposal. On rejection, planner revises up to configured retries.
    4) If planner/critic loop cannot produce an accepted proposal, fallback to
       planner's best candidate.
    """

    VERSION = '1.0'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._oracle_planner: OraclePlanner | None = None
        self._proposal_critic: OracleProposalCritic | None = None
        self._components_initialized = False

        self.triad_log: list[dict] = []

        self._num_candidates = max(
            int(os.environ.get('BLINDED_DEBUGGER_NUM_CANDIDATES', '3')),
            1,
        )
        self._planner_max_retries = max(
            int(os.environ.get('ORACLE_PLANNER_MAX_RETRIES', '2')),
            0,
        )

    def step(self, state: 'State') -> 'Action':
        if self.pending_actions:
            return self.pending_actions.popleft()

        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events
            case Condensation(action=condensation_action):
                return condensation_action

        if not self._components_initialized:
            self._init_components(state)

        initial_user_message = self._get_initial_user_message(state.history)
        messages = self._get_messages(condensed_history, initial_user_message)

        params: dict = {
            'messages': messages,
            'tools': check_tools(self.tools, self.llm.config),
            'extra_body': {
                'metadata': state.to_llm_metadata(
                    model_name=self.llm.config.model,
                    agent_name=self.name,
                )
            },
        }

        step_index = state.iteration_flag.current_value
        full_history_text = self._render_history_text_full(state.history)

        candidate_responses = []
        candidate_texts: list[str] = []

        for candidate_index in range(self._num_candidates):
            response = self.llm.completion(**params)
            text = self._extract_response_text(response)
            candidate_responses.append(response)
            candidate_texts.append(text)

            entry = {
                'step_index': step_index,
                'event': 'debugger_candidate',
                'candidate_index': candidate_index,
                'response_text': text,
            }
            self.triad_log.append(entry)
            _append_triage_entry(entry)

        chosen_response = None
        planner_feedback = ''
        planner_best_idx = 0

        for planner_attempt in range(self._planner_max_retries + 1):
            decision = self._plan_next_response(
                step_index=step_index,
                history_text=full_history_text,
                candidates=candidate_texts,
                feedback=planner_feedback,
                attempt=planner_attempt,
            )
            planner_best_idx = decision.best_candidate_index

            planner_entry = {
                'step_index': step_index,
                'event': 'oracle_planner_decision',
                'attempt': planner_attempt,
                **decision.to_dict(),
            }
            self.triad_log.append(planner_entry)
            _append_triage_entry(planner_entry)

            if decision.decision == 'candidate':
                idx = decision.chosen_candidate_index
                if idx is None or idx < 0 or idx >= len(candidate_responses):
                    idx = planner_best_idx
                chosen_response = candidate_responses[idx]
                break

            if self._proposal_critic is None:
                chosen_response = self._materialize_planner_proposal(
                    base_messages=params['messages'],
                    planner_proposal=decision.proposal_response_text,
                    state=state,
                )
                break

            validation = self._proposal_critic.validate(
                step_index=step_index,
                history_text=full_history_text,
                proposal_response_text=decision.proposal_response_text,
                attempt=planner_attempt,
            )

            critic_entry = {
                'step_index': step_index,
                'event': 'proposal_critic_validation',
                'attempt': planner_attempt,
                **validation.to_dict(),
            }
            self.triad_log.append(critic_entry)
            _append_triage_entry(critic_entry)

            if validation.valid:
                chosen_response = self._materialize_planner_proposal(
                    base_messages=params['messages'],
                    planner_proposal=decision.proposal_response_text,
                    state=state,
                )
                break

            if planner_attempt < self._planner_max_retries:
                planner_feedback = validation.feedback_message or validation.reason
                continue

            logger.warning(
                '[OracleTriadCodeActAgent] Planner proposal retries exhausted at step '
                f'{step_index}; falling back to best candidate index {planner_best_idx}.'
            )
            fallback_idx = planner_best_idx
            if fallback_idx < 0 or fallback_idx >= len(candidate_responses):
                fallback_idx = 0
            chosen_response = candidate_responses[fallback_idx]
            break

        if chosen_response is None:
            logger.warning(
                f'[OracleTriadCodeActAgent] No response selected at step {step_index}; using candidate 0 fallback.'
            )
            chosen_response = candidate_responses[0]

        actions = self.response_to_actions(chosen_response)
        for action in actions:
            self.pending_actions.append(action)
        return self.pending_actions.popleft()

    def _init_components(self, state: 'State') -> None:
        self._components_initialized = True

        try:
            raw_instruction = self._get_raw_initial_instruction(state.history)
            stripped_issue_text = self._strip_oracle_blocks(raw_instruction)
            public_issue_text = self._extract_issue_description(stripped_issue_text)
            if not public_issue_text:
                public_issue_text = stripped_issue_text
            oracle_context = self._load_oracle_context()
            tool_descriptions = self._build_tool_descriptions()

            self._oracle_planner = OraclePlanner.from_env(
                issue_text=public_issue_text,
                oracle_context=oracle_context,
                tool_descriptions=tool_descriptions,
            )
            self._proposal_critic = OracleProposalCritic.from_env(
                issue_text=public_issue_text,
            )
        except Exception as exc:
            logger.warning(
                '[OracleTriadCodeActAgent] Failed to initialize planner/critic components: '
                f'{exc}. Falling back to candidate-only mode.'
            )
            self._oracle_planner = None
            self._proposal_critic = None

    def _plan_next_response(
        self,
        step_index: int,
        history_text: str,
        candidates: list[str],
        feedback: str,
        attempt: int,
    ) -> PlannerDecision:
        if self._oracle_planner is None:
            return PlannerDecision(
                step_index=step_index,
                decision='candidate',
                best_candidate_index=0,
                chosen_candidate_index=0,
                reason='Oracle planner disabled; using candidate 0.',
                proposal_response_text='',
                raw_planner_response='',
            )

        decision = self._oracle_planner.plan(
            step_index=step_index,
            history_text=history_text,
            candidates=candidates,
            planner_feedback=feedback,
            attempt=attempt,
        )

        if decision.best_candidate_index < 0 or decision.best_candidate_index >= len(candidates):
            decision.best_candidate_index = 0

        if decision.decision == 'candidate':
            if decision.chosen_candidate_index is None:
                decision.chosen_candidate_index = decision.best_candidate_index
            elif decision.chosen_candidate_index < 0 or decision.chosen_candidate_index >= len(candidates):
                decision.chosen_candidate_index = decision.best_candidate_index

        return decision

    def _materialize_planner_proposal(
        self,
        base_messages: list[Message],
        planner_proposal: str,
        state: 'State',
    ):
        proposal_messages = self._inject_planner_guidance(base_messages, planner_proposal)
        proposal_params = {
            'messages': proposal_messages,
            'tools': check_tools(self.tools, self.llm.config),
            'extra_body': {
                'metadata': state.to_llm_metadata(
                    model_name=self.llm.config.model,
                    agent_name=self.name,
                )
            },
        }
        response = self.llm.completion(**proposal_params)

        # When the debugger LLM produces tool calls, it often leaves content
        # empty (e.g. content=[] or content=None).  The downstream
        # response_to_actions() would use content as the "thought" for the
        # action, so an empty content means the planner's reasoning text is
        # lost.  Fix: if content is empty but tool_calls exist, extract the
        # REASONING part from the original proposal and inject it.
        if response and response.choices:
            msg = response.choices[0].message
            has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
            content_empty = (
                msg.content is None
                or msg.content == ''
                or (isinstance(msg.content, list) and len(msg.content) == 0)
            )
            if has_tool_calls and content_empty:
                reasoning = self._extract_reasoning_from_proposal(planner_proposal)
                if reasoning:
                    msg.content = reasoning

        return response

    @staticmethod
    def _extract_reasoning_from_proposal(proposal: str) -> str:
        """Extract the REASONING portion from a planner proposal.

        The proposal format is:
            <reasoning text>
            [TOOL CALL] tool_name({...})

        Returns the text before the first [TOOL CALL] marker, stripped.
        If [TOOL CALL] is at the very start, returns empty string.
        """
        marker = '[TOOL CALL]'
        idx = proposal.find(marker)
        if idx >= 0:
            return proposal[:idx].strip()
        # No marker found — the whole proposal is reasoning
        return proposal.strip()

    @staticmethod
    def _inject_planner_guidance(messages: list[Message], guidance: str) -> list[Message]:
        message = Message(
            role='user',
            content=[
                TextContent(
                    text=(
                        '[ORACLE PLANNER GUIDANCE - APPROVED BY BLINDED CRITIC]\n\n'
                        'The following guidance contains two parts:\n'
                        '1. REASONING: suggested thought process / analysis text for your response.\n'
                        '2. TOOL CALL: suggested tool invocation(s) to execute.\n\n'
                        'You MUST:\n'
                        '- Incorporate the reasoning text naturally as your own thought process.\n'
                        '- Execute the suggested tool call(s) with the indicated parameters.\n'
                        '- Only use information already observed in this session.\n'
                        '- Maintain your normal response format (reasoning text followed by tool call).\n\n'
                        '--- GUIDANCE START ---\n'
                        f'{guidance}\n'
                        '--- GUIDANCE END ---'
                    )
                )
            ],
        )
        return list(messages) + [message]

    def _load_oracle_context(self) -> str:
        context_path = os.environ.get('ORACLE_PLANNER_CONTEXT_PATH', '').strip()
        if not context_path:
            return 'No oracle context file was provided.'

        try:
            with open(context_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning(
                f'[OracleTriadCodeActAgent] Failed to load oracle context from {context_path}: {exc}'
            )
            return f'Failed to load oracle context ({exc}).'

        patch = str(payload.get('patch', '') or '')
        test_patch = str(payload.get('test_patch', '') or '')
        issue_understanding = str(payload.get('issue_understanding', '') or '')

        parts = [
            '## Golden Patch',
            patch,
            '',
            '## Golden Test Patch',
            test_patch,
            '',
            '## Issue Understanding Package',
            issue_understanding if issue_understanding else '(not provided)',
        ]
        return '\n'.join(parts)

    def _build_tool_descriptions(self) -> str:
        """Build a human-readable summary of available tools for the planner."""
        lines: list[str] = []
        for tool in self.tools:
            fn = tool.get('function', {})
            name = fn.get('name', '(unknown)')
            desc = fn.get('description', '')
            params = fn.get('parameters', {}).get('properties', {})
            required = fn.get('parameters', {}).get('required', [])

            param_parts: list[str] = []
            for pname, pinfo in params.items():
                ptype = pinfo.get('type', 'any')
                pdesc = pinfo.get('description', '')
                req_marker = ' (REQUIRED)' if pname in required else ''
                # Truncate long param descriptions for planner readability
                if len(pdesc) > 200:
                    pdesc = pdesc[:200] + '...'
                param_parts.append(f'    - {pname} ({ptype}{req_marker}): {pdesc}')

            lines.append(f'### {name}')
            if desc:
                # Truncate long tool descriptions
                short_desc = desc if len(desc) <= 500 else desc[:500] + '...'
                lines.append(short_desc)
            if param_parts:
                lines.append('  Parameters:')
                lines.extend(param_parts)
            lines.append('')

        return '\n'.join(lines) if lines else '(no tools available)'

    @staticmethod
    def _get_raw_initial_instruction(history: list[Event]) -> str:
        for event in history:
            if isinstance(event, MessageAction) and event.source == 'user':
                return event.content
        return ''

    @staticmethod
    def _strip_oracle_blocks(text: str) -> str:
        text = _REFERENCE_BLOCK_RE.sub('', text)
        text = _ISSUE_UNDERSTANDING_BLOCK_RE.sub('', text)
        return text.strip()

    @staticmethod
    def _extract_issue_description(text: str) -> str:
        match = _ISSUE_DESCRIPTION_BLOCK_RE.search(text)
        if not match:
            return ''
        return match.group(0).strip()

    @staticmethod
    def _strip_issue_description_block(text: str) -> str:
        return _ISSUE_DESCRIPTION_BLOCK_RE.sub('', text).strip()

    @staticmethod
    def _extract_response_text(response) -> str:
        if not response or not response.choices:
            return '(empty response)'

        choice = response.choices[0]
        msg = choice.message
        parts: list[str] = []

        content = msg.content
        if content:
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        parts.append(block.get('text', ''))

        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                fn = tc.function
                parts.append(f'[TOOL CALL] {fn.name}({fn.arguments})')

        return '\n'.join(parts) if parts else '(empty response)'

    @staticmethod
    def _render_history_text_full(events: list[Event]) -> str:
        from openhands.events.action.action import Action
        from openhands.events.action.agent import AgentThinkAction
        from openhands.events.action.message import SystemMessageAction
        from openhands.events.action import CmdRunAction, FileEditAction, FileReadAction
        from openhands.events.observation import (
            CmdOutputObservation,
            FileEditObservation,
            FileReadObservation,
        )

        files_read: list[str] = []
        files_edited: list[str] = []
        commands_run: list[str] = []
        system_messages: list[tuple[int, str]] = []
        user_messages: list[tuple[int, str]] = []

        def _truncate_text(text: str, limit: int = 6000) -> str:
            if len(text) <= limit:
                return text
            return text[:limit] + '\n...<truncated>...'

        for event in events:
            if isinstance(event, FileReadAction):
                if event.path not in files_read:
                    files_read.append(event.path)
            elif isinstance(event, FileEditAction):
                if event.path not in files_edited:
                    files_edited.append(event.path)
            elif isinstance(event, CmdRunAction):
                commands_run.append(event.command)

        for i, event in enumerate(events):
            if isinstance(event, SystemMessageAction):
                system_messages.append((i, event.content))
            if isinstance(event, MessageAction) and event.source == 'user':
                user_messages.append((i, event.content))

        index_lines: list[str] = ['=== SESSION INDEX (full history) ===']
        if system_messages or user_messages:
            index_lines.append('Base instruction events (hard requirements):')
            if system_messages:
                system_idx, system_content = system_messages[0]
                index_lines.append(
                    f'  - History Event {system_idx} OPENHANDS SYSTEM MESSAGE:'
                )
                for line in system_content.splitlines() or ['']:
                    index_lines.append(f'      {line}')
            if user_messages:
                user_idx, user_content = user_messages[0]
                user_content = OracleTriadCodeActAgent._strip_issue_description_block(
                    user_content
                )
                index_lines.append(
                    f'  - History Event {user_idx} USER MESSAGE (SWE-bench workflow):'
                )
                for line in user_content.splitlines() or ['']:
                    index_lines.append(f'      {line}')
        if files_read:
            index_lines.append('Files READ this session:')
            for path in files_read:
                index_lines.append(f'  - {path}')
        if files_edited:
            index_lines.append('Files EDITED this session:')
            for path in files_edited:
                index_lines.append(f'  - {path}')
        if commands_run:
            index_lines.append('Commands RUN this session:')
            for command in commands_run:
                index_lines.append(f'  - {command}')

        index_lines.append('=== END SESSION INDEX ===')
        index_lines.append('')

        lines: list[str] = []
        for i, event in enumerate(events):
            if isinstance(event, SystemMessageAction):
                continue
            if isinstance(event, MessageAction) and event.source == 'user':
                continue
            if isinstance(event, MessageAction) and event.source == 'agent':
                lines.append(f'[Event {i}] AGENT MESSAGE: {event.content}')
            elif isinstance(event, AgentThinkAction):
                lines.append(f'[Event {i}] AGENT THOUGHT: {event.thought}')
            elif isinstance(event, CmdRunAction):
                if event.thought:
                    lines.append(f'[Event {i}] AGENT THOUGHT: {event.thought}')
                lines.append(f'[Event {i}] RUN COMMAND: {event.command}')
            elif isinstance(event, FileReadAction):
                if event.thought:
                    lines.append(f'[Event {i}] AGENT THOUGHT: {event.thought}')
                lines.append(f'[Event {i}] READ FILE: {event.path}')
            elif isinstance(event, FileEditAction):
                if event.thought:
                    lines.append(f'[Event {i}] AGENT THOUGHT: {event.thought}')
                lines.append(f'[Event {i}] EDIT FILE: {event.path} -- {event.content}')
            elif isinstance(event, CmdOutputObservation):
                lines.append(f'[Event {i}] OBS (exit={event.exit_code}): {event.content}')
            elif isinstance(event, FileReadObservation):
                lines.append(f'[Event {i}] OBS (file read): {event.content}')
            elif isinstance(event, FileEditObservation):
                lines.append(f'[Event {i}] OBS (file edit): {event.content}')
            elif isinstance(event, Observation):
                lines.append(f'[Event {i}] OBS: {str(event.content)}')
            elif isinstance(event, Action):
                action_name = type(event).__name__
                action_message = getattr(event, 'message', '') or ''
                lines.append(f'[Event {i}] ACTION ({action_name}): {action_message}')
            else:
                lines.append(f'[Event {i}] EVENT ({type(event).__name__}): {str(event)}')

        body = '\n'.join(lines) if lines else '(no prior interactions)'
        return '\n'.join(index_lines) + body
