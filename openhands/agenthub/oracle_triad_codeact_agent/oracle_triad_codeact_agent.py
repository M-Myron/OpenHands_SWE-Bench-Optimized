from __future__ import annotations

import json
import os
import re
import time
from typing import TYPE_CHECKING

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.agenthub.oracle_triad_codeact_agent.history_memory import (
    StructuredHistoryMemory,
)
from openhands.agenthub.oracle_triad_codeact_agent.oracle_planner import (
    OraclePlanner,
    PlannerDecision,
    ReactFactTracker,
)
from openhands.agenthub.oracle_triad_codeact_agent.triad_config import TriadConfig
from openhands.agenthub.oracle_triad_codeact_agent.proposal_critic import (
    OracleProposalCritic,
    ProposalValidationResult,
)
from openhands.agenthub.oracle_triad_codeact_agent.verifier import (
    HistoryGroundedVerifier,
    VerificationVerdict,
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
    """Triad agent with blinded debugger, oracle planner, and proposal validator.

    Flow per step:
    1) Blinded debugger (primary agent LLM) generates N candidate responses.
    2) Oracle planner (oracle-aware LLM) inspects full interaction history and
       candidates, then either selects one candidate or proposes a revised
       response while still being non-leaky and history-grounded.
    3) If planner proposes a revised response, the proposal validator checks it.
       On rejection, planner revises up to configured retries.
    4) If planner/validator loop cannot produce an accepted proposal, fallback to
       planner's best candidate.

    The validator backend is selected via the ``PROPOSAL_VALIDATOR`` env var:

    - ``verifier`` (default) — ``HistoryGroundedVerifier``: 4-stage neuro-symbolic
      pipeline (claim extraction → retrieval → symbolic rules → verdict synthesis).
    - ``critic`` — ``OracleProposalCritic``: legacy one-shot LLM critic.
    - ``none`` — skip proposal validation entirely.
    """

    VERSION = '1.0'

    # Accepted values for PROPOSAL_VALIDATOR env var
    _VALID_VALIDATORS = ('verifier', 'critic', 'none')

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._oracle_planner: OraclePlanner | None = None
        self._proposal_critic: OracleProposalCritic | None = None
        self._verifier: HistoryGroundedVerifier | None = None
        self._components_initialized = False

        # Load config (YAML file or defaults) and export to env vars
        self._triad_config = TriadConfig.load()
        self._triad_config.export_to_env()

        # Read validator selection ------------------------------------------
        # PROPOSAL_VALIDATOR: 'verifier' | 'critic' | 'none'  (default: verifier)
        # Legacy compat: USE_LEGACY_CRITIC=1 is equivalent to PROPOSAL_VALIDATOR=critic
        raw = os.environ.get('PROPOSAL_VALIDATOR', '').strip().lower()
        if raw and raw in self._VALID_VALIDATORS:
            self._validator_mode: str = raw
        elif os.environ.get('USE_LEGACY_CRITIC', '0') == '1':
            self._validator_mode = 'critic'
        else:
            self._validator_mode = 'verifier'

        self.triad_log: list[dict] = []

        self._num_candidates = max(
            int(os.environ.get('BLINDED_DEBUGGER_NUM_CANDIDATES', '3')),
            1,
        )
        self._planner_max_retries = max(
            int(os.environ.get('ORACLE_PLANNER_MAX_RETRIES', '2')),
            0,
        )
        # -1 = full history (no windowing), positive int = last N action steps
        _hw = int(os.environ.get('ORACLE_PLANNER_HISTORY_WINDOW', '5'))
        self._planner_history_window: int = _hw if _hw >= 1 else -1

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
        # Windowed history for the planner (keeps summary + last N steps; -1 = full)
        if self._planner_history_window < 0:
            planner_history_text = full_history_text
        else:
            planner_history_text = self._render_history_text_windowed(
                state.history, window=self._planner_history_window,
            )

        # ---- Timing instrumentation ----
        step_t0 = time.monotonic()
        timing: dict[str, float] = {}
        llm_calls: dict[str, int] = {}

        candidate_responses = []
        candidate_texts: list[str] = []

        t0 = time.monotonic()
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
        timing['candidates'] = time.monotonic() - t0
        llm_calls['candidates'] = self._num_candidates

        chosen_response = None
        planner_feedback = ''
        planner_best_idx = 0
        planner_llm_calls = 0
        verifier_llm_calls = 0
        planner_time = 0.0
        verifier_time = 0.0
        materialization_time = 0.0

        for planner_attempt in range(self._planner_max_retries + 1):
            t0 = time.monotonic()
            decision = self._plan_next_response(
                step_index=step_index,
                history_text=planner_history_text,
                candidates=candidate_texts,
                feedback=planner_feedback,
                attempt=planner_attempt,
            )
            planner_time += time.monotonic() - t0
            planner_llm_calls += 1  # at least 1 per attempt (may have JSON retries)
            planner_best_idx = decision.best_candidate_index

            planner_entry = {
                'step_index': step_index,
                'event': 'oracle_planner_decision',
                'attempt': planner_attempt,
                **decision.to_dict(),
            }
            self.triad_log.append(planner_entry)
            _append_triage_entry(planner_entry)

            # Get preconditions for referenced facts (for critic validation)
            fact_preconditions: list[dict] = []
            referenced_ids = decision.referenced_fact_ids
            if referenced_ids and self._oracle_planner and self._oracle_planner.react_fact_tracker:
                fact_preconditions = self._oracle_planner.react_fact_tracker.get_preconditions_for_facts(
                    referenced_ids
                )

            if decision.decision == 'candidate':
                idx = decision.chosen_candidate_index
                if idx is None or idx < 0 or idx >= len(candidate_responses):
                    idx = planner_best_idx
                chosen_response = candidate_responses[idx]
                # Mark referenced facts as used on accepted decision
                if referenced_ids and self._oracle_planner and self._oracle_planner.react_fact_tracker:
                    self._oracle_planner.react_fact_tracker.mark_facts_used(referenced_ids, step_index=step_index)
                self._oracle_planner.record_accepted_decision(decision)
                break

            if self._proposal_critic is None and self._verifier is None:
                t_m = time.monotonic()
                chosen_response = self._materialize_planner_proposal(
                    base_messages=params['messages'],
                    planner_proposal=decision.proposal_response_text,
                    state=state,
                )
                materialization_time += time.monotonic() - t_m
                # Mark referenced facts as used on accepted proposal (no critic/verifier)
                if referenced_ids and self._oracle_planner and self._oracle_planner.react_fact_tracker:
                    self._oracle_planner.react_fact_tracker.mark_facts_used(referenced_ids, step_index=step_index)
                self._oracle_planner.record_accepted_decision(decision)
                break

            # --- Validate proposal via verifier or legacy critic -----------
            validation_valid = False
            validation_feedback = ''
            validation_reason = ''

            if self._verifier is not None:
                t_v = time.monotonic()
                history_memory = StructuredHistoryMemory.from_events(state.history)
                verdict = self._verifier.verify(
                    step_index=step_index,
                    proposal_text=decision.proposal_response_text,
                    history_memory=history_memory,
                    fact_preconditions=fact_preconditions if fact_preconditions else None,
                    attempt=planner_attempt,
                )
                verifier_time += time.monotonic() - t_v
                # Count verifier LLM calls from the timing log
                vtiming = getattr(verdict, '_timing', None)
                if vtiming and isinstance(vtiming, dict):
                    verifier_llm_calls += vtiming.get('llm_calls', 0)
                else:
                    # Estimate: extraction(1) + resolution(0-1) + synthesis(0-1)
                    verifier_llm_calls += 1  # at least extraction

                verifier_entry = {
                    'step_index': step_index,
                    'event': 'verifier_verdict',
                    'attempt': planner_attempt,
                    **verdict.to_dict(),
                }
                self.triad_log.append(verifier_entry)
                _append_triage_entry(verifier_entry)

                validation_valid = verdict.valid
                validation_feedback = verdict.feedback_message
                validation_reason = verdict.reason

            elif self._proposal_critic is not None:
                validation = self._proposal_critic.validate(
                    step_index=step_index,
                    history_text=full_history_text,
                    proposal_response_text=decision.proposal_response_text,
                    attempt=planner_attempt,
                    fact_preconditions=fact_preconditions if fact_preconditions else None,
                )

                critic_entry = {
                    'step_index': step_index,
                    'event': 'proposal_critic_validation',
                    'attempt': planner_attempt,
                    **validation.to_dict(),
                }
                self.triad_log.append(critic_entry)
                _append_triage_entry(critic_entry)

                validation_valid = validation.valid
                validation_feedback = validation.feedback_message or validation.reason
                validation_reason = validation.reason

            if validation_valid:
                t_m = time.monotonic()
                chosen_response = self._materialize_planner_proposal(
                    base_messages=params['messages'],
                    planner_proposal=decision.proposal_response_text,
                    state=state,
                )
                materialization_time += time.monotonic() - t_m
                # Mark referenced facts as used on accepted proposal
                if referenced_ids and self._oracle_planner and self._oracle_planner.react_fact_tracker:
                    self._oracle_planner.react_fact_tracker.mark_facts_used(referenced_ids, step_index=step_index)
                self._oracle_planner.record_accepted_decision(decision)
                break

            if planner_attempt < self._planner_max_retries:
                # Include the rejected proposal text so the planner knows what was rejected
                rejected_proposal = decision.proposal_response_text
                planner_feedback = (
                    f'## Your Previous Proposal (rejected)\n'
                    f'{rejected_proposal}\n\n'
                    f'## Rejection Reason\n'
                    f'{validation_feedback or validation_reason}'
                )
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

        # Log fact usage summary
        if self._oracle_planner and self._oracle_planner.react_fact_tracker and self._oracle_planner.react_fact_tracker.has_facts:
            usage_summary = self._oracle_planner.react_fact_tracker.get_usage_summary()
            fact_entry = {
                'step_index': step_index,
                'event': 'react_fact_usage_summary',
                **usage_summary,
            }
            self.triad_log.append(fact_entry)
            _append_triage_entry(fact_entry)

        actions = self.response_to_actions(chosen_response)
        for action in actions:
            self.pending_actions.append(action)

        # ---- Timing summary ----
        timing['planner'] = planner_time
        timing['verifier'] = verifier_time
        timing['materialization'] = materialization_time
        timing['step_total'] = time.monotonic() - step_t0
        llm_calls['planner'] = planner_llm_calls
        llm_calls['verifier'] = verifier_llm_calls
        llm_calls['materialization'] = 1 if materialization_time > 0 else 0
        llm_calls['total'] = sum(llm_calls.values())

        timing_entry = {
            'step_index': step_index,
            'event': 'step_timing',
            'timing_seconds': {k: round(v, 2) for k, v in timing.items()},
            'llm_calls': llm_calls,
        }
        self.triad_log.append(timing_entry)
        _append_triage_entry(timing_entry)

        logger.info(
            f'[OracleTriadCodeActAgent] Step {step_index} timing: '
            f'candidates={timing["candidates"]:.1f}s ({llm_calls["candidates"]} calls), '
            f'planner={timing["planner"]:.1f}s ({llm_calls["planner"]} calls), '
            f'verifier={timing["verifier"]:.1f}s ({llm_calls["verifier"]} calls), '
            f'materialization={timing["materialization"]:.1f}s, '
            f'total={timing["step_total"]:.1f}s ({llm_calls["total"]} LLM calls)'
        )
        return self.pending_actions.popleft()

    def _init_components(self, state: 'State') -> None:
        self._components_initialized = True

        try:
            raw_instruction = self._get_raw_initial_instruction(state.history)
            stripped_issue_text = self._strip_oracle_blocks(raw_instruction)
            public_issue_text = self._extract_issue_description(stripped_issue_text)
            if not public_issue_text:
                public_issue_text = stripped_issue_text
            oracle_context, react_facts_data = self._load_oracle_context()
            tool_descriptions = self._build_tool_descriptions()

            react_fact_tracker = ReactFactTracker(react_facts_data)

            self._oracle_planner = OraclePlanner.from_env(
                issue_text=public_issue_text,
                oracle_context=oracle_context,
                tool_descriptions=tool_descriptions,
                react_fact_tracker=react_fact_tracker,
                prompt_config=self._triad_config.planner_prompt,
            )

            # ---- initialise selected proposal validator ------------------
            if self._validator_mode == 'critic':
                self._proposal_critic = OracleProposalCritic.from_env(
                    issue_text=public_issue_text,
                )
                logger.info(
                    '[OracleTriadCodeActAgent] Proposal validator: OracleProposalCritic (legacy).'
                )
            elif self._validator_mode == 'verifier':
                self._verifier = HistoryGroundedVerifier.from_env(
                    issue_text=public_issue_text,
                )
                if self._verifier is None:
                    logger.warning(
                        '[OracleTriadCodeActAgent] Verifier init returned None; '
                        'falling back to OracleProposalCritic.'
                    )
                    self._proposal_critic = OracleProposalCritic.from_env(
                        issue_text=public_issue_text,
                    )
                else:
                    logger.info(
                        '[OracleTriadCodeActAgent] Proposal validator: HistoryGroundedVerifier.'
                    )
            else:
                # 'none' — no validation
                logger.info(
                    '[OracleTriadCodeActAgent] Proposal validator: none (validation disabled).'
                )
        except Exception as exc:
            logger.warning(
                '[OracleTriadCodeActAgent] Failed to initialize planner/critic components: '
                f'{exc}. Falling back to candidate-only mode.'
            )
            self._oracle_planner = None
            self._proposal_critic = None
            self._verifier = None

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

        if decision.best_candidate_index is None or decision.best_candidate_index < 0 or decision.best_candidate_index >= len(candidates):
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

    def _load_oracle_context(self) -> tuple[str, dict | None]:
        """Load oracle context and return (context_text, react_facts_data).

        Which sections are included is controlled by ``self._triad_config.oracle_context``.
        """
        context_path = os.environ.get('ORACLE_PLANNER_CONTEXT_PATH', '').strip()
        if not context_path:
            return 'No oracle context file was provided.', None

        try:
            with open(context_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning(
                f'[OracleTriadCodeActAgent] Failed to load oracle context from {context_path}: {exc}'
            )
            return f'Failed to load oracle context ({exc}).', None

        ctx_cfg = self._triad_config.oracle_context

        patch = str(payload.get('patch', '') or '')
        test_patch = str(payload.get('test_patch', '') or '')
        issue_understanding = str(payload.get('issue_understanding', '') or '')
        deep_analysis = str(payload.get('deep_analysis', '') or '')
        react_facts = payload.get('react_facts', None) if ctx_cfg.include_react_facts else None

        parts: list[str] = []

        if ctx_cfg.include_golden_patch:
            parts.extend(['## Golden Patch', patch, ''])
        else:
            parts.extend(['## Golden Patch', '(disabled by config)', ''])

        if ctx_cfg.include_golden_test_patch:
            parts.extend(['## Golden Test Patch', test_patch, ''])
        else:
            parts.extend(['## Golden Test Patch', '(disabled by config)', ''])

        if ctx_cfg.include_issue_understanding:
            parts.extend([
                '## Issue Understanding Package',
                issue_understanding if issue_understanding else '(not provided)',
            ])
        else:
            parts.extend(['## Issue Understanding Package', '(disabled by config)'])

        if ctx_cfg.include_deep_analysis and deep_analysis:
            parts.extend([
                '',
                '## Deep Root-Cause Analysis',
                'The following is a detailed precomputed analysis of the bug, including '
                'step-by-step traces from user action to crash, the fixed chain, '
                'knowledge requirements, and investigation logs. Use this to guide '
                'the debugger toward the correct fix without leaking specifics.',
                '',
                deep_analysis,
            ])

        return '\n'.join(parts), react_facts

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

        index_lines: list[str] = ['=== SESSION SUMMARY ===']
        if system_messages:
            index_lines.append(f'  - Event 0: OpenHands system instruction (debugger capabilities)')
        if user_messages:
            index_lines.append(f'  - Event 1: SWE-bench task + 8-phase workflow (issue description provided separately)')
        if files_read:
            index_lines.append('Files read:')
            for path in files_read:
                index_lines.append(f'  - {path}')
        if files_edited:
            index_lines.append('Files edited:')
            for path in files_edited:
                index_lines.append(f'  - {path}')
        if commands_run:
            index_lines.append('Commands run:')
            for command in commands_run:
                index_lines.append(f'  - {command}')

        index_lines.append('=== END SESSION SUMMARY ===')
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

    @staticmethod
    def _render_history_text_windowed(events: list[Event], window: int = 5) -> str:
        """Render history text with only the last `window` action-observation pairs.

        Keeps the full SESSION SUMMARY (file/command inventories) but truncates
        the event body to only recent steps, saving tokens in the planner prompt.
        """
        from openhands.events.action.action import Action
        from openhands.events.action.agent import AgentThinkAction
        from openhands.events.action.message import SystemMessageAction
        from openhands.events.action import CmdRunAction, FileEditAction, FileReadAction
        from openhands.events.observation import (
            CmdOutputObservation,
            FileEditObservation,
            FileReadObservation,
        )

        # --- Build the same SESSION SUMMARY as the full version ---
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
                commands_run.append(event.command)

        index_lines: list[str] = ['=== SESSION SUMMARY ===']
        index_lines.append(f'  - Event 0: OpenHands system instruction (debugger capabilities)')
        index_lines.append(f'  - Event 1: SWE-bench task + 8-phase workflow (issue description provided separately)')
        if files_read:
            index_lines.append('Files read:')
            for path in files_read:
                index_lines.append(f'  - {path}')
        if files_edited:
            index_lines.append('Files edited:')
            for path in files_edited:
                index_lines.append(f'  - {path}')
        if commands_run:
            index_lines.append('Commands run:')
            for command in commands_run:
                index_lines.append(f'  - {command}')
        index_lines.append('=== END SESSION SUMMARY ===')
        index_lines.append('')

        # --- Count action events (non-system, non-user) to find the window ---
        # An "action step" is a non-system, non-user Action event.
        action_indices: list[int] = []
        for i, event in enumerate(events):
            if isinstance(event, SystemMessageAction):
                continue
            if isinstance(event, MessageAction) and event.source == 'user':
                continue
            if isinstance(event, Action):
                action_indices.append(i)

        # Determine the starting event index for the window
        if len(action_indices) <= window:
            # Show all events
            window_start_event = 0
        else:
            # Show only events from the `window`-th-last action onwards
            window_start_event = action_indices[-window]

        # --- Render only events within the window ---
        total_events = len(events)
        skipped = sum(1 for i, e in enumerate(events)
                      if i < window_start_event
                      and not isinstance(e, SystemMessageAction)
                      and not (isinstance(e, MessageAction) and e.source == 'user'))

        lines: list[str] = []
        if skipped > 0:
            lines.append(f'... ({skipped} earlier events omitted, see SESSION SUMMARY above) ...')
            lines.append('')

        for i, event in enumerate(events):
            if i < window_start_event:
                continue
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

        body = '\n'.join(lines) if lines else '(no prior interactions)'
        return '\n'.join(index_lines) + body
