"""Oracle Guided CodeAct Agent.

Three-component orchestration: Blinded Solver (candidates) → Oracle Planner
(select / revise / rewrite) → Hybrid Critic (neural + symbolic validation).
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import TYPE_CHECKING
from uuid import uuid4

from litellm import ModelResponse
from litellm.types.utils import Choices, Message as LitellmMessage, Usage

import re as _re

from openhands.llm.llm import LLM_RETRY_EXCEPTIONS

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.agenthub.oracle_guided_codeact_agent.fact_tracker import (
    STAGE_ANALYSIS_PLANNING,
    STAGE_EXPLORATION,
    STAGE_FINISH,
    STAGE_IMPLEMENTATION_VERIFICATION,
    STAGE_REPRODUCTION,
    FactTracker,
)
from openhands.agenthub.oracle_guided_codeact_agent.guided_config import GuidedConfig
from openhands.agenthub.oracle_guided_codeact_agent.hybrid_critic import (
    CriticResult,
    HybridCritic,
)
from openhands.agenthub.oracle_guided_codeact_agent.oracle_planner import (
    OraclePlanner,
    PlannerDecision,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import AgentFinishAction, MessageAction
from openhands.events.action.message import SystemMessageAction
from openhands.events.event import Event
from openhands.events.observation import Observation
from openhands.llm.llm_utils import check_tools
from openhands.memory.condenser.condenser import Condensation, View

if TYPE_CHECKING:
    from openhands.controller.state.state import State
    from openhands.events.action import Action


# ---------------------------------------------------------------------------
# Per-process triage log (mirrors the triad module pattern)
# ---------------------------------------------------------------------------

_TRIAGE_LOG: list[dict] = []
_TRIAGE_PID: int | None = None


def _ensure_triage_file() -> str:
    global _TRIAGE_PID
    pid = os.getpid()
    _TRIAGE_PID = pid
    return f'/tmp/oracle_guided_{pid}.jsonl'


def _append_triage(entry: dict) -> None:
    _TRIAGE_LOG.append(entry)
    path = _ensure_triage_file()
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception:
        pass


def clear_triage_log() -> None:
    global _TRIAGE_LOG
    _TRIAGE_LOG = []
    path = _ensure_triage_file()
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def read_and_clear_triage_log() -> list[dict]:
    global _TRIAGE_LOG
    log = list(_TRIAGE_LOG)
    clear_triage_log()
    return log


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class OracleGuidedCodeActAgent(CodeActAgent):
    """Oracle-guided agent with blinded solver, oracle planner, and hybrid critic."""

    VERSION = '2.0'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Load config
        self._guided_config = GuidedConfig.load()
        self._guided_config.export_to_env()

        # Component state — lazy init on first step
        self._oracle_planner: OraclePlanner | None = None
        self._hybrid_critic: HybridCritic | None = None
        self._fact_tracker: FactTracker | None = None
        self._components_initialized: bool = False

        # Runtime config from env (after export_to_env)
        self._num_candidates = max(1, int(os.environ.get('GUIDED_NUM_CANDIDATES', '1')))
        self._planner_max_retries = max(0, int(os.environ.get('GUIDED_PLANNER_MAX_RETRIES', '2')))
        self._gate_max_retries = max(0, int(os.environ.get('GUIDED_GATE_MAX_RETRIES', '2')))
        self._oracle_start_step = max(0, int(os.environ.get('GUIDED_ORACLE_START_STEP', '0')))
        self._oracle_auto_activate = os.environ.get('GUIDED_ORACLE_AUTO_ACTIVATE', '0') == '1'
        self._oracle_auto_activate_fallback_step = max(
            1, int(os.environ.get('GUIDED_ORACLE_AUTO_ACTIVATE_FALLBACK_STEP', '5'))
        )
        self._oracle_activated = False  # tracks whether auto-activation has fired
        self._history_near_window = int(os.environ.get('GUIDED_PLANNER_HISTORY_NEAR_WINDOW', '5'))

        self.guided_log: list[dict] = []

    # ------------------------------------------------------------------
    # Oracle auto-activation
    # ------------------------------------------------------------------

    _PHASE3_PATTERN = _re.compile(r'##\s*Phase\s*3', _re.IGNORECASE)

    def _should_oracle_be_active(
        self, step_index: int, history_events: list['Event'],
    ) -> bool:
        """Determine whether the oracle planner should be active this step.

        When ``oracle_auto_activate`` is disabled, falls back to the simple
        ``oracle_start_step`` threshold.

        When enabled:
        1. If already activated in a prior step, stay active.
        2. Scan the solver's reasoning in history for ``## Phase 3`` header.
           If found, activate.
        3. If ``step_index >= oracle_auto_activate_fallback_step``, activate.
        4. Otherwise, not yet active — let the solver run unguided.
        """
        if not self._oracle_auto_activate:
            return step_index >= self._oracle_start_step

        if self._oracle_activated:
            return True

        # Scan history for Phase 3 header in agent messages
        for event in history_events:
            if not hasattr(event, 'source') or getattr(event, 'source', '') != 'agent':
                continue
            text = ''
            if isinstance(event, MessageAction):
                text = event.content or ''
            elif hasattr(event, 'thought'):
                text = getattr(event, 'thought', '') or ''
            if text and self._PHASE3_PATTERN.search(text):
                self._oracle_activated = True
                logger.info(
                    f'[OracleGuided] Auto-activation triggered: '
                    f'Phase 3 header detected in history at step {step_index}'
                )
                _append_triage({
                    'step_index': step_index,
                    'event': 'oracle_auto_activated',
                    'trigger': 'phase3_header',
                })
                return True

        # Fallback step threshold
        if step_index >= self._oracle_auto_activate_fallback_step:
            self._oracle_activated = True
            logger.info(
                f'[OracleGuided] Auto-activation triggered: '
                f'fallback step {self._oracle_auto_activate_fallback_step} '
                f'reached at step {step_index}'
            )
            _append_triage({
                'step_index': step_index,
                'event': 'oracle_auto_activated',
                'trigger': 'fallback_step',
            })
            return True

        logger.info(
            f'[OracleGuided] Step {step_index}: oracle not yet active '
            f'(auto-activate waiting for Phase 3 header or step '
            f'{self._oracle_auto_activate_fallback_step})'
        )
        return False

    # ------------------------------------------------------------------
    # Phase gate checking
    # ------------------------------------------------------------------

    @staticmethod
    def _has_file_creation(response_text: str) -> bool:
        """Return True if the response contains a file-creation action."""
        if _re.search(
            r'str_replace_editor.*"command"\s*:\s*"create"',
            response_text,
            _re.IGNORECASE | _re.DOTALL,
        ):
            return True
        if _re.search(r'>\s*\S+\.py\b', response_text):
            return True
        return False

    @staticmethod
    def _has_code_modification(response_text: str) -> bool:
        """Return True if the response contains a code-modifying action.

        Detects ``str_replace_editor`` with ``str_replace`` or ``insert``
        commands, or ``sed -i`` in bash.
        """
        if _re.search(
            r'str_replace_editor.*"command"\s*:\s*"(?:str_replace|insert)"',
            response_text,
            _re.IGNORECASE | _re.DOTALL,
        ):
            return True
        if _re.search(r'\bsed\s+-i\b', response_text):
            return True
        return False

    # Gate block reason types
    _GATE_STAGE_CONSTRAINT = 'stage_constraint'
    _GATE_OK = 'ok'

    def _check_phase_gate(
        self, response_text: str, decision_type: str,
    ) -> tuple[bool, str, str]:
        """Check if response violates the current stage's constraints.

        Returns ``(ok, feedback, reason_type)`` where *reason_type* is one of:
        - ``'ok'``              — no gate violation
        - ``'stage_constraint'`` — action blocked by stage rules

        The caller uses *reason_type* to decide fallback strategy when
        retries are exhausted.
        """
        if not self._fact_tracker or not self._fact_tracker.has_facts:
            return True, '', self._GATE_OK

        stage = self._fact_tracker.get_current_stage()
        source = 'The selected candidate' if decision_type == 'select' else 'Your revision'

        if stage == STAGE_EXPLORATION:
            if self._has_file_creation(response_text):
                breakdown = self._fact_tracker.get_unexplored_fact_breakdown()
                total = breakdown['total']
                avail = breakdown['count_available']

                # Build actionable feedback — don't dump the full fact list
                # (the planner already sees categorized facts in its prompt).
                parts = [
                    f'{source} tries to create a file, '
                    f'but there are still {total} unexplored investigation '
                    f'fact(s) ({avail} currently available to target).',
                    '',
                    'Before transitioning to file creation:',
                    '1. If any facts from the "Unlocker Satisfied — Needs '
                    'Articulation" section remain, articulate them first — '
                    'add reasoning that interprets the finding, then mark as used.',
                    '2. If facts in "Available" sections remain, guide the solver '
                    'to perform the next unlocker action.',
                    '3. If ALL remaining facts are clearly unrelated to the '
                    'reported issue, you may proceed with file creation and the '
                    'system will verify the early phase transition.',
                    '',
                    'Do NOT create files yet — instead continue with the '
                    'appropriate action above.',
                ]
                feedback = '\n'.join(parts)
                logger.info(
                    f'[OracleGuided] Stage gate blocked file creation in '
                    f'EXPLORATION stage (decision={decision_type})'
                )
                return False, feedback, self._GATE_STAGE_CONSTRAINT

            if self._has_code_modification(response_text):
                feedback = (
                    f'{source} tries to modify code, '
                    f'but the solver is still in the exploration stage.\n\n'
                    f'The solver must first explore all facts, then reproduce, '
                    f'then analyze before implementing fixes.'
                )
                logger.info(
                    f'[OracleGuided] Stage gate blocked code modification in '
                    f'EXPLORATION stage (decision={decision_type})'
                )
                return False, feedback, self._GATE_STAGE_CONSTRAINT

        elif stage == STAGE_REPRODUCTION:
            if self._has_code_modification(response_text):
                feedback = (
                    f'{source} tries to modify code, '
                    f'but the solver should be creating a reproduction script first.\n\n'
                    f'Guide the solver to Phase 4: TEST CREATION — create and run '
                    f'a reproduction script before any analysis or implementation.'
                )
                logger.info(
                    f'[OracleGuided] Stage gate blocked code modification in '
                    f'REPRODUCTION stage (decision={decision_type})'
                )
                return False, feedback, self._GATE_STAGE_CONSTRAINT

        elif stage == STAGE_ANALYSIS_PLANNING:
            if self._has_code_modification(response_text):
                feedback = (
                    f'{source} tries to modify code, '
                    f'but the solver has not completed analysis and planning.\n\n'
                    f'Guide the solver to Phase 5: FIX ANALYSIS — use the `think` '
                    f'tool to analyze the root cause and create a fix plan first.'
                )
                logger.info(
                    f'[OracleGuided] Stage gate blocked code modification in '
                    f'ANALYSIS_PLANNING stage (decision={decision_type})'
                )
                return False, feedback, self._GATE_STAGE_CONSTRAINT

        return True, '', self._GATE_OK

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, state: 'State') -> 'Action':
        # Pending actions from prior step
        if self.pending_actions:
            return self.pending_actions.popleft()

        # Exit check
        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        # Condense history
        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events
            case Condensation(action=condensation_action):
                return condensation_action

        # Lazy init components
        if not self._components_initialized:
            self._init_components(state)

        step_index = len([
            e for e in state.history
            if hasattr(e, 'source') and e.source == 'agent'
            and not isinstance(e, Observation)
        ])

        # Build messages for blinded solver (standard CodeActAgent path)
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

        # =============================================================
        # Phase 1: Generate N candidate responses (Blinded Solver)
        # =============================================================
        t0 = time.monotonic()
        candidate_responses: list[ModelResponse] = []
        candidate_texts: list[str] = []
        last_candidate_error: Exception | None = None

        _CANDIDATE_TRANSIENT_RETRIES = int(os.environ.get('GUIDED_TRANSIENT_RETRIES', '5'))
        _CANDIDATE_RETRY_BASE_WAIT = int(os.environ.get('GUIDED_RETRY_BASE_WAIT', '10'))

        for ci in range(self._num_candidates):
            succeeded = False
            for retry_i in range(_CANDIDATE_TRANSIENT_RETRIES):
                try:
                    response = self.llm.completion(**params)
                    text = self._extract_response_text(response)
                    candidate_responses.append(response)
                    candidate_texts.append(text)
                    _append_triage({
                        'step_index': step_index,
                        'event': 'solver_candidate',
                        'candidate_index': ci,
                        'response_text': text[:2000],
                    })
                    succeeded = True
                    break
                except LLM_RETRY_EXCEPTIONS as exc:
                    # Transient error (network, rate-limit, timeout) —
                    # tenacity already exhausted its retries.  Give the
                    # endpoint a moment and retry the candidate.
                    if retry_i < _CANDIDATE_TRANSIENT_RETRIES - 1:
                        wait = _CANDIDATE_RETRY_BASE_WAIT * (2 ** retry_i)
                        logger.warning(
                            f'[OracleGuided] Candidate {ci} transient error '
                            f'(attempt {retry_i + 1}/{_CANDIDATE_TRANSIENT_RETRIES}), '
                            f'retrying in {wait}s: {exc}'
                        )
                        time.sleep(wait)
                    else:
                        logger.warning(
                            f'[OracleGuided] Candidate {ci} generation failed '
                            f'after {_CANDIDATE_TRANSIENT_RETRIES} transient retries: {exc}'
                        )
                        last_candidate_error = exc
                except Exception as exc:
                    # Non-transient error — no point retrying
                    logger.warning(f'[OracleGuided] Candidate {ci} generation failed (non-transient): {exc}')
                    last_candidate_error = exc
                    break

        t_candidates = time.monotonic() - t0

        if not candidate_responses:
            # Re-raise the last LLM error so the controller sees it as an
            # ERROR state (not FINISHED), which triggers the retry mechanism
            # in the evaluation harness.
            if last_candidate_error is not None:
                logger.error(
                    f'[OracleGuided] All {self._num_candidates} candidates '
                    f'failed. Re-raising last error for retry.'
                )
                raise last_candidate_error
            logger.error('[OracleGuided] All candidates failed (no error captured). Returning finish action.')
            return AgentFinishAction()

        # =============================================================
        # Phase 2: Oracle Planner + Critic Loop
        # =============================================================
        chosen_response: ModelResponse | None = None
        planner_feedback = ''
        t1 = time.monotonic()
        # Per-attempt timing lists
        planner_times: list[float] = []
        critic_times: list[float] = []
        final_attempt = 0
        final_decision_type = 'select'

        if self._oracle_planner is not None and self._should_oracle_be_active(step_index, state.history):
            # Build history texts.
            # In the FINISH stage, use full untruncated history so the
            # oracle can produce a meaningful summary of the entire
            # solving trajectory.
            current_stage = (
                self._fact_tracker.get_current_stage()
                if self._fact_tracker else STAGE_EXPLORATION
            )
            if current_stage == STAGE_FINISH:
                history_text_planner = self._render_full_history(state.history)
            else:
                history_text_planner = self._render_windowed_history(
                    state.history, self._history_near_window,
                    include_system_instruction=self._guided_config.planner.include_system_instruction,
                )
            history_text_critic = self._render_full_history(state.history)

            # Total budget = gate_max + critic_max retries (worst case both
            # types fire on different attempts).  Each type has its own
            # exhaustion counter so they don't eat each other's budget.
            max_total_attempts = self._gate_max_retries + self._planner_max_retries + 1
            gate_blocks = 0
            critic_rejects = 0

            for attempt in range(max_total_attempts):
                final_attempt = attempt

                t_plan_start = time.monotonic()
                decision = self._oracle_planner.plan(
                    step_index=step_index,
                    history_text=history_text_planner,
                    candidates=candidate_texts,
                    feedback=planner_feedback,
                    attempt=attempt,
                )
                planner_times.append(round(time.monotonic() - t_plan_start, 2))

                final_decision_type = decision.decision

                _append_triage({
                    'step_index': step_index,
                    'event': 'oracle_planner_decision',
                    'attempt': attempt,
                    'stage': self._oracle_planner.current_stage,
                    'decision': decision.decision,
                    'candidate_index': decision.candidate_index,
                    'facts_used': decision.facts_used,
                    'next_target_fact': decision.next_target_fact,
                    'reason': decision.reason,
                    'response_content': decision.response_content[:1000] if decision.response_content else '',
                })

                # ---- Phase gate check (before critic) ----
                if decision.decision == 'select':
                    text_to_check = candidate_texts[
                        min(decision.candidate_index, len(candidate_texts) - 1)
                    ]
                else:
                    text_to_check = (
                        (decision.response_content or '')
                        + ' '
                        + ' '.join(
                            json.dumps(tc) for tc in (decision.response_tool_calls or [])
                        )
                    )

                gate_ok, gate_feedback, gate_reason = self._check_phase_gate(
                    text_to_check, decision.decision,
                )
                if not gate_ok:
                    # Stage gate blocked — check if this is exploration
                    # trying to transition with unused facts.
                    # Ask the critic if the remaining facts are relevant.
                    stage = self._fact_tracker.get_current_stage()
                    relevance_result: dict | None = None
                    if (
                        stage == STAGE_EXPLORATION
                        and self._hybrid_critic is not None
                    ):
                        unused = self._fact_tracker.get_unused_fact_statements()
                        if unused:
                            relevance_result = self._hybrid_critic.check_facts_relevance(
                                unused,
                                step_index=step_index,
                                attempt=attempt,
                            )
                            if relevance_result['all_irrelevant']:
                                skipped = self._fact_tracker.skip_remaining_facts()
                                _append_triage({
                                    'step_index': step_index,
                                    'event': 'exploration_exit_approved',
                                    'skipped_facts': skipped,
                                })
                                # Stage has advanced — re-query the planner
                                # in the new stage so it sees the artifact
                                # content (e.g. reproduction script).
                                # Do NOT accept the current response which
                                # was generated by the old stage prompt.
                                new_stage = self._oracle_planner.current_stage
                                logger.info(
                                    f'[OracleGuided] Facts skipped as '
                                    f'irrelevant. Re-querying planner in '
                                    f'new stage: {new_stage}'
                                )
                                history_text_planner = self._render_windowed_history(
                                    state.history, self._history_near_window,
                                    include_system_instruction=self._guided_config.planner.include_system_instruction,
                                )
                                t_requery_start = time.monotonic()
                                new_decision = self._oracle_planner.plan(
                                    step_index=step_index,
                                    history_text=history_text_planner,
                                    candidates=candidate_texts,
                                    feedback='',
                                    attempt=attempt + 1,
                                )
                                planner_times.append(
                                    round(time.monotonic() - t_requery_start, 2)
                                )
                                final_decision_type = new_decision.decision
                                _append_triage({
                                    'step_index': step_index,
                                    'event': 'post_skip_requery',
                                    'new_stage': new_stage,
                                    'decision': new_decision.decision,
                                    'facts_used': new_decision.facts_used,
                                })
                                if new_decision.decision == 'select':
                                    chosen_response = candidate_responses[
                                        min(new_decision.candidate_index,
                                            len(candidate_responses) - 1)
                                    ]
                                else:
                                    chosen_response = self._build_synthetic_response(
                                        new_decision
                                    )
                                if new_decision.facts_used and self._fact_tracker:
                                    self._fact_tracker.mark_used(
                                        new_decision.facts_used, step_index
                                    )
                                self._oracle_planner.record_accepted_decision(
                                    new_decision
                                )
                                # Skip the rest of the retry loop
                                gate_ok = True
                                break

                if not gate_ok:
                    gate_blocks += 1

                    # Augment gate feedback with per-fact relevance info
                    # from the critic, if available.
                    if relevance_result and relevance_result.get('relevant_ids'):
                        rel_ids = relevance_result['relevant_ids']
                        irrel_ids = relevance_result.get('irrelevant_ids', [])
                        rel_reason = relevance_result.get('reason', '')
                        extra = (
                            f'\n\nThe critic identified which facts are '
                            f'relevant to the issue:\n'
                            f'  Relevant (must explore): {rel_ids}\n'
                        )
                        if irrel_ids:
                            extra += (
                                f'  Irrelevant (can skip): {irrel_ids}\n'
                            )
                        if rel_reason:
                            extra += f'  Reason: {rel_reason}\n'
                        extra += (
                            '\nFocus on the relevant facts. If you '
                            'articulate or explore them, the gate will clear.'
                        )
                        gate_feedback = gate_feedback + extra

                    _append_triage({
                        'step_index': step_index,
                        'event': 'stage_gate_block',
                        'attempt': attempt,
                        'gate_blocks': gate_blocks,
                        'stage': self._oracle_planner.current_stage,
                        'gate_reason': gate_reason,
                        'decision': decision.decision,
                        'feedback': gate_feedback[:1000],
                        'relevance_result': relevance_result,
                        'facts_used_in_rejected': decision.facts_used,
                    })

                    # If the rejected decision tried to mark facts as used,
                    # honour that — the articulation in response_content is
                    # valid even though the tool call (file creation) is not.
                    # Marking them now may advance the stage so the next
                    # attempt's gate check passes naturally.
                    stage_before_mark = (
                        self._fact_tracker.get_current_stage()
                        if self._fact_tracker else None
                    )
                    if decision.facts_used and self._fact_tracker:
                        self._fact_tracker.mark_used(
                            decision.facts_used, step_index
                        )
                        self._oracle_planner.record_accepted_decision(decision)
                        logger.info(
                            f'[OracleGuided] Marked facts from gate-rejected '
                            f'decision: {decision.facts_used}'
                        )
                        new_stage_after_mark = self._fact_tracker.get_current_stage()
                        _append_triage({
                            'step_index': step_index,
                            'event': 'gate_rejected_facts_marked',
                            'facts_used': decision.facts_used,
                            'stage_before': stage_before_mark,
                            'new_stage': new_stage_after_mark,
                        })

                        # If marking facts caused a stage transition,
                        # re-query the planner in the new stage immediately
                        # (the current response was generated for the old
                        # stage and shouldn't be used).
                        if new_stage_after_mark != stage_before_mark:
                            logger.info(
                                f'[OracleGuided] Stage advanced from '
                                f'{stage_before_mark} to '
                                f'{new_stage_after_mark} after marking '
                                f'facts. Re-querying planner.'
                            )
                            history_text_planner = self._render_windowed_history(
                                state.history, self._history_near_window,
                                include_system_instruction=self._guided_config.planner.include_system_instruction,
                            )
                            t_requery_start = time.monotonic()
                            new_decision = self._oracle_planner.plan(
                                step_index=step_index,
                                history_text=history_text_planner,
                                candidates=candidate_texts,
                                feedback='',
                                attempt=attempt + 1,
                            )
                            planner_times.append(
                                round(time.monotonic() - t_requery_start, 2)
                            )
                            final_decision_type = new_decision.decision
                            _append_triage({
                                'step_index': step_index,
                                'event': 'post_mark_stage_requery',
                                'old_stage': stage_before_mark,
                                'new_stage': new_stage_after_mark,
                                'decision': new_decision.decision,
                                'facts_used': new_decision.facts_used,
                            })
                            if new_decision.decision == 'select':
                                chosen_response = candidate_responses[
                                    min(new_decision.candidate_index,
                                        len(candidate_responses) - 1)
                                ]
                            else:
                                chosen_response = self._build_synthetic_response(
                                    new_decision
                                )
                            if new_decision.facts_used and self._fact_tracker:
                                self._fact_tracker.mark_used(
                                    new_decision.facts_used, step_index
                                )
                            self._oracle_planner.record_accepted_decision(
                                new_decision
                            )
                            gate_ok = True
                            break

                        # Stage didn't change — just add feedback for retry
                        gate_feedback += (
                            '\n\nYour articulation of findings was accepted '
                            'and the facts were marked as used. However, '
                            'do NOT combine articulation with file creation '
                            'or code modification in the same response. '
                            'Just produce the articulation reasoning '
                            '(without any tool call that creates files or '
                            'modifies code). The stage will advance and '
                            'those actions will be handled in the next step.'
                        )

                    # Include the full rejected response so planner sees
                    # what was blocked.
                    planner_feedback = (
                        gate_feedback
                        + '\n\n## YOUR REJECTED RESPONSE (do NOT repeat this):\n'
                        + text_to_check
                    )
                    if gate_blocks > self._gate_max_retries:
                        if gate_reason == self._GATE_STAGE_CONSTRAINT:
                            # Stage constraint — force stage transition by
                            # skipping remaining facts, then re-query the
                            # planner with the NEW stage prompt (which
                            # includes artifact content for the next stage).
                            if self._fact_tracker and not self._fact_tracker.all_facts_used():
                                skipped = self._fact_tracker.skip_remaining_facts()
                                logger.info(
                                    f'[OracleGuided] Gate retries exhausted '
                                    f'(stage constraint) at step {step_index}. '
                                    f'Forcing stage transition, skipped facts: '
                                    f'{skipped}'
                                )
                                _append_triage({
                                    'step_index': step_index,
                                    'event': 'forced_stage_transition',
                                    'skipped_facts': skipped,
                                })
                            else:
                                logger.warning(
                                    f'[OracleGuided] Gate retries exhausted '
                                    f'(stage constraint) at step {step_index} '
                                    f'but all facts already used.'
                                )

                            # Re-query planner in the new stage context
                            new_stage = self._oracle_planner.current_stage
                            logger.info(
                                f'[OracleGuided] Re-querying planner in new '
                                f'stage: {new_stage}'
                            )
                            # Rebuild history for the new stage prompt
                            history_text_planner = self._render_windowed_history(
                                state.history, self._history_near_window,
                                include_system_instruction=self._guided_config.planner.include_system_instruction,
                            )
                            t_requery_start = time.monotonic()
                            new_decision = self._oracle_planner.plan(
                                step_index=step_index,
                                history_text=history_text_planner,
                                candidates=candidate_texts,
                                feedback='',
                                attempt=attempt + 1,
                            )
                            planner_times.append(
                                round(time.monotonic() - t_requery_start, 2)
                            )
                            final_decision_type = new_decision.decision
                            _append_triage({
                                'step_index': step_index,
                                'event': 'post_transition_requery',
                                'new_stage': new_stage,
                                'decision': new_decision.decision,
                                'facts_used': new_decision.facts_used,
                            })

                            if new_decision.decision == 'select':
                                chosen_response = candidate_responses[
                                    min(new_decision.candidate_index,
                                        len(candidate_responses) - 1)
                                ]
                            else:
                                chosen_response = self._build_synthetic_response(
                                    new_decision
                                )
                            if new_decision.facts_used and self._fact_tracker:
                                self._fact_tracker.mark_used(
                                    new_decision.facts_used, step_index
                                )
                            self._oracle_planner.record_accepted_decision(
                                new_decision
                            )
                        else:
                            # Unknown gate reason — fall back to raw candidate
                            logger.warning(
                                f'[OracleGuided] Gate retries exhausted '
                                f'(reason={gate_reason}) at step {step_index}. '
                                f'Falling back to raw candidate.'
                            )
                            chosen_response = candidate_responses[
                                min(decision.candidate_index, len(candidate_responses) - 1)
                            ]
                        break
                    continue  # retry with gate feedback

                # ---- Accept select ----
                if decision.decision == 'select':
                    idx = decision.candidate_index
                    if 0 <= idx < len(candidate_responses):
                        chosen_response = candidate_responses[idx]
                    else:
                        chosen_response = candidate_responses[0]

                    # Mark facts used
                    if decision.facts_used and self._fact_tracker:
                        self._fact_tracker.mark_used(decision.facts_used, step_index)
                    # Record accepted decision for cross-step continuity
                    self._oracle_planner.record_accepted_decision(decision)
                    break

                else:  # revise or rewrite
                    # Build the oracle response text for critic
                    oracle_response_text = self._format_oracle_response_text(decision)

                    # Validate with critic
                    if self._hybrid_critic is not None:
                        t_critic_start = time.monotonic()
                        critic_result = self._hybrid_critic.validate(
                            step_index=step_index,
                            best_candidate_text=candidate_texts[0],
                            oracle_response_text=oracle_response_text,
                            history_text=history_text_critic,
                            attempt=attempt,
                        )
                        critic_times.append(round(time.monotonic() - t_critic_start, 2))

                        _append_triage({
                            'step_index': step_index,
                            'event': 'critic_result',
                            'attempt': attempt,
                            'valid': critic_result.valid,
                            'neural_valid': critic_result.neural_valid,
                            'neural_reasons': critic_result.neural_reasons,
                            'symbolic_failures': len(critic_result.symbolic_failures),
                            'rechecked_failures': len(critic_result.rechecked_failures),
                        })

                        if critic_result.valid:
                            chosen_response = self._build_synthetic_response(decision)
                            if decision.facts_used and self._fact_tracker:
                                self._fact_tracker.mark_used(decision.facts_used, step_index)
                            self._oracle_planner.record_accepted_decision(decision)
                            break
                        else:
                            # Rejected — build feedback for planner retry
                            critic_rejects += 1
                            planner_feedback = critic_result.feedback_message
                            if critic_rejects > self._planner_max_retries:
                                # Exhausted retries — accept planner's last
                                # response rather than falling back to the raw
                                # candidate (which may be stage-violating).
                                logger.warning(
                                    f'[OracleGuided] Critic retries exhausted at '
                                    f'step {step_index}. Accepting planner\'s last '
                                    f'response (decision={decision.decision}).'
                                )
                                chosen_response = self._build_synthetic_response(decision)
                                if decision.facts_used and self._fact_tracker:
                                    self._fact_tracker.mark_used(decision.facts_used, step_index)
                                self._oracle_planner.record_accepted_decision(decision)
                                break
                    else:
                        # No critic — accept directly
                        chosen_response = self._build_synthetic_response(decision)
                        if decision.facts_used and self._fact_tracker:
                            self._fact_tracker.mark_used(decision.facts_used, step_index)
                        self._oracle_planner.record_accepted_decision(decision)
                        break

        t_planner_critic = time.monotonic() - t1

        # Fallback
        if chosen_response is None:
            chosen_response = candidate_responses[0]

        # =============================================================
        # Phase 3: Fact usage summary + timing log
        # =============================================================
        if self._fact_tracker and self._fact_tracker.has_facts:
            summary = self._fact_tracker.get_usage_summary()
            _append_triage({
                'step_index': step_index,
                'event': 'fact_usage_summary',
                **summary,
            })
            # Check if all edits+validations just completed — start
            # the finish extension countdown.
            self._fact_tracker.check_and_set_impl_complete(step_index)

        total_time = time.monotonic() - t0
        t_planner_total = sum(planner_times)
        t_critic_total = sum(critic_times)
        _append_triage({
            'step_index': step_index,
            'event': 'step_timing',
            'candidates_sec': round(t_candidates, 2),
            'planner_sec': round(t_planner_total, 2),
            'critic_sec': round(t_critic_total, 2),
            'planner_critic_sec': round(t_planner_critic, 2),
            'total_sec': round(total_time, 2),
            'num_candidates': len(candidate_responses),
            'attempts': final_attempt + 1,
            'decision': final_decision_type,
            'planner_times': planner_times,
            'critic_times': critic_times,
        })

        # Build detailed timing string
        timing_parts = [
            f'candidates={t_candidates:.1f}s ({len(candidate_responses)})',
        ]
        # Add stage info if available
        if self._oracle_planner is not None:
            timing_parts.append(f'stage={self._oracle_planner.current_stage}')
        if len(planner_times) == 1 and not critic_times:
            # Simple case: one planner call, select decision, no critic
            timing_parts.append(f'planner={planner_times[0]:.1f}s ({final_decision_type})')
        else:
            # Multi-attempt or critic involved
            for i, pt in enumerate(planner_times):
                timing_parts.append(f'plan[{i}]={pt:.1f}s')
            for i, ct in enumerate(critic_times):
                timing_parts.append(f'critic[{i}]={ct:.1f}s')
            timing_parts.append(f'decision={final_decision_type}')
            if final_attempt > 0:
                timing_parts.append(f'attempts={final_attempt + 1}')

        timing_parts.append(f'total={total_time:.1f}s')
        logger.info(f'[OracleGuided] Step {step_index}: {", ".join(timing_parts)}')

        # =============================================================
        # Phase 4: Convert to actions
        # =============================================================
        actions = self.response_to_actions(chosen_response)
        for action in actions:
            self.pending_actions.append(action)
        return self.pending_actions.popleft()

    # ------------------------------------------------------------------
    # Component initialization
    # ------------------------------------------------------------------

    def _init_components(self, state: 'State') -> None:
        """Lazy-initialize oracle planner, critic, and fact tracker."""
        self._components_initialized = True

        # Load oracle context
        oracle_context_text, react_facts_data = self._load_oracle_context()
        if not oracle_context_text and react_facts_data is None:
            logger.warning('[OracleGuided] No oracle context found. Running as plain CodeActAgent.')
            return

        # Extract issue text and tool instructions from history
        issue_text = ''
        tool_instructions = ''
        for event in state.history:
            if isinstance(event, MessageAction) and event.source == 'user':
                issue_text = self._extract_issue_description(event.content)
                break

        # Get tool instructions from system message
        for event in state.history:
            if hasattr(event, 'content') and hasattr(event, 'source'):
                if getattr(event, 'source', '') == 'system' or (
                    hasattr(event, '__class__') and 'SystemMessage' in event.__class__.__name__
                ):
                    tool_instructions = self._extract_tool_instructions(event.content)
                    break

        # If no system message found, build tool descriptions from self.tools
        if not tool_instructions:
            tool_instructions = self._build_tool_descriptions()

        # Initialize fact tracker
        if react_facts_data:
            self._fact_tracker = FactTracker(react_facts_data)
            self._fact_tracker.set_finish_extension_budget(
                self._guided_config.agent.finish_extension_steps
            )
            logger.info(
                f'[OracleGuided] FactTracker initialized with '
                f'{len(self._fact_tracker.node_ids)} nodes. '
                f'Finish extension budget: '
                f'{self._guided_config.agent.finish_extension_steps} steps.'
            )

        # Initialize oracle planner
        self._oracle_planner = OraclePlanner.from_env(
            issue_text=issue_text,
            oracle_context=oracle_context_text,
            tool_instructions=tool_instructions,
            fact_tracker=self._fact_tracker,
        )
        if self._oracle_planner:
            logger.info('[OracleGuided] Oracle planner initialized.')

        # Initialize hybrid critic
        self._hybrid_critic = HybridCritic.from_env(
            issue_text=issue_text,
        )
        if self._hybrid_critic:
            logger.info('[OracleGuided] Hybrid critic initialized.')

    # ------------------------------------------------------------------
    # Oracle context loading
    # ------------------------------------------------------------------

    def _load_oracle_context(self) -> tuple[str, dict | None]:
        """Load oracle context from JSON file. Returns (context_text, react_facts_data)."""
        context_path = os.environ.get('ORACLE_GUIDED_CONTEXT_PATH', '').strip()
        if not context_path or not os.path.isfile(context_path):
            return '', None

        try:
            with open(context_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f'[OracleGuided] Failed to load oracle context: {exc}')
            return '', None

        cfg = self._guided_config.oracle_context
        parts: list[str] = []

        if cfg.include_golden_patch and data.get('patch'):
            parts.append(f'## Golden Patch\n```diff\n{data["patch"]}\n```')
        if cfg.include_golden_test_patch and data.get('test_patch'):
            parts.append(f'## Golden Test Patch\n```diff\n{data["test_patch"]}\n```')
        if cfg.include_issue_understanding and data.get('issue_understanding'):
            parts.append(f'## Issue Understanding\n{data["issue_understanding"]}')

        react_facts_data = None
        if cfg.include_react_facts and data.get('react_facts'):
            react_facts_data = data['react_facts']

        context_text = '\n\n'.join(parts)
        logger.info(f'[OracleGuided] Oracle context loaded ({len(context_text)} chars).')
        return context_text, react_facts_data

    # ------------------------------------------------------------------
    # History rendering
    # ------------------------------------------------------------------

    def _render_windowed_history(
        self, events: list[Event], near_window: int,
        include_system_instruction: bool = True,
    ) -> str:
        """Render history with far events (action only) and near events (action + observation)."""
        # Collect action-observation pairs, skipping the first agent
        # MessageAction which is the system-instruction echo (already
        # rendered in the SYSTEM INSTRUCTION section).
        steps: list[dict] = []
        current_action: dict | None = None
        skipped_first_agent_msg = False

        for event in events:
            if isinstance(event, Observation):
                if current_action is not None:
                    current_action['observation'] = self._format_observation(event)
                continue

            if hasattr(event, 'source') and event.source == 'agent':
                if not skipped_first_agent_msg and isinstance(event, (MessageAction, SystemMessageAction)):
                    skipped_first_agent_msg = True
                    continue
                if current_action is not None:
                    steps.append(current_action)
                current_action = {
                    'action': self._format_action(event),
                    'observation': '',
                }

        if current_action is not None:
            steps.append(current_action)

        if not steps:
            return '(No interaction history yet.)'

        # Determine cutoff
        if near_window <= 0 or near_window >= len(steps):
            cutoff = 0  # Show everything as near
        else:
            cutoff = len(steps) - near_window

        parts: list[str] = []

        # Session index: system message + initial user message
        if include_system_instruction:
            for event in events:
                if hasattr(event, '__class__') and 'SystemMessage' in event.__class__.__name__:
                    content = getattr(event, 'content', '')
                    if content:
                        parts.append(f'=== SYSTEM INSTRUCTION ===\n{content[:3000]}...\n')
                    break

        for event in events:
            if isinstance(event, MessageAction) and event.source == 'user':
                parts.append(f'=== INITIAL USER MESSAGE ===\n{event.content}\n')
                break

        parts.append('=== INTERACTION STEPS ===\n')

        for i, step in enumerate(steps):
            if i < cutoff:
                # Far: action only
                parts.append(f'Step {i}: [ACTION] {step["action"]}')
            else:
                # Near: action + observation
                parts.append(f'Step {i}: [ACTION] {step["action"]}')
                if step['observation']:
                    parts.append(f'  [OBSERVATION] {step["observation"]}')

        return '\n'.join(parts)

    def _render_full_history(self, events: list[Event]) -> str:
        """Render full untruncated history for critic."""
        parts: list[str] = []
        for i, event in enumerate(events):
            if isinstance(event, Observation):
                obs_text = self._format_observation(event)
                parts.append(f'[Event {i} - OBSERVATION] {obs_text}')
            elif hasattr(event, 'source'):
                action_text = self._format_action(event)
                source = getattr(event, 'source', 'unknown')
                parts.append(f'[Event {i} - {source.upper()} ACTION] {action_text}')

        return '\n\n'.join(parts) if parts else '(No history yet.)'

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_action(self, event: Event) -> str:
        """Format an action event into readable text for the planner prompt."""
        from openhands.events.action.commands import CmdRunAction, IPythonRunCellAction
        from openhands.events.action.files import (
            FileEditAction,
            FileReadAction,
            FileWriteAction,
        )

        if isinstance(event, MessageAction):
            return f'Message: {event.content}'

        thought = getattr(event, 'thought', '') or ''
        parts = []
        if thought:
            parts.append(f'Thought: {thought}')

        if isinstance(event, CmdRunAction):
            parts.append(f'Command: {event.command}')
        elif isinstance(event, FileReadAction):
            vr = ''
            if event.view_range:
                vr = f' (lines {event.view_range[0]}-{event.view_range[1]})'
            elif event.start != 0 or event.end != -1:
                vr = f' (lines {event.start}-{event.end})'
            parts.append(f'File: {event.path}{vr}')
        elif isinstance(event, FileEditAction):
            cmd = event.command or 'edit'
            if cmd == 'view':
                vr = ''
                if getattr(event, 'view_range', None):
                    vr = f' (lines {event.view_range[0]}-{event.view_range[1]})'
                parts.append(f'File: {event.path}{vr}')
            elif cmd == 'create':
                text_preview = (event.file_text or '')[:500]
                parts.append(f'Create file: {event.path}\nContent: {text_preview}')
            elif cmd == 'str_replace':
                parts.append(
                    f'Edit file: {event.path}\n'
                    f'Old: {(event.old_str or "")[:300]}\n'
                    f'New: {(event.new_str or "")[:300]}'
                )
            elif cmd == 'insert':
                parts.append(
                    f'Insert in file: {event.path} after line {event.insert_line}\n'
                    f'Text: {(event.new_str or "")[:300]}'
                )
            else:
                parts.append(f'{cmd}: {event.path}')
        elif isinstance(event, FileWriteAction):
            parts.append(f'Write file: {event.path}\nContent: {event.content[:500]}')
        elif isinstance(event, IPythonRunCellAction):
            parts.append(f'IPython: {event.code[:1000]}')
        elif hasattr(event, 'command'):
            parts.append(f'Command: {event.command}')
        elif hasattr(event, 'path'):
            path = getattr(event, 'path', '')
            parts.append(f'File: {path}')

        cls_name = event.__class__.__name__
        if not parts:
            parts.append(f'{cls_name}: {str(event)[:1000]}')

        return ' | '.join(parts)

    def _format_observation(self, event: Observation) -> str:
        """Format an observation event into readable text."""
        content = getattr(event, 'content', '')
        if not content:
            content = str(event)
        return content

    def _extract_response_text(self, response: ModelResponse) -> str:
        """Extract readable text from a ModelResponse.

        Tool calls are formatted as JSON objects matching the planner's
        output schema: ``{"name": "...", "arguments": {...}}``.
        """
        if not response or not response.choices:
            return ''
        msg = response.choices[0].message
        parts = []
        content = msg.content
        if isinstance(content, str) and content:
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    parts.append(item['text'])

        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_obj = {
                    'name': tc.function.name,
                    'arguments': json.loads(tc.function.arguments)
                    if isinstance(tc.function.arguments, str)
                    else tc.function.arguments,
                }
                parts.append(f'[TOOL CALL] {json.dumps(tc_obj)}')

        return '\n'.join(parts)

    def _extract_issue_description(self, text: str) -> str:
        """Extract issue description from initial user message."""
        # Try to find <issue_description> block
        match = re.search(
            r'<issue_description>(.*?)</issue_description>', text, re.DOTALL
        )
        if match:
            return match.group(1).strip()
        # Fallback: return first 5000 chars
        return text[:5000]

    def _extract_tool_instructions(self, system_message: str) -> str:
        """Extract tool usage instructions from system message."""
        if not system_message:
            return ''
        # Try to find tool-related sections
        # Look for sections describing tool usage
        patterns = [
            r'(## Tools.*?)(?=\n## |\Z)',
            r'(# Tools.*?)(?=\n# |\Z)',
            r'(You have access to.*?tools.*?)(?=\n## |\n# |\Z)',
        ]
        for pat in patterns:
            match = re.search(pat, system_message, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        # Fallback: return summary from self.tools
        return self._build_tool_descriptions()

    def _build_tool_descriptions(self) -> str:
        """Build human-readable tool descriptions from self.tools."""
        if not self.tools:
            return '(No tools available.)'

        parts = ['## Available Tools\n']
        for tool in self.tools:
            if not isinstance(tool, dict):
                continue
            func = tool.get('function', {})
            name = func.get('name', '?')
            desc = func.get('description', '')
            if len(desc) > 500:
                desc = desc[:500] + '...'
            parts.append(f'### {name}')
            parts.append(f'{desc}\n')

            params = func.get('parameters', {})
            props = params.get('properties', {})
            required = set(params.get('required', []))
            if props:
                parts.append('Parameters:')
                for pname, pinfo in props.items():
                    ptype = pinfo.get('type', '?')
                    pdesc = pinfo.get('description', '')
                    if len(pdesc) > 200:
                        pdesc = pdesc[:200] + '...'
                    req = ' (required)' if pname in required else ''
                    parts.append(f'  - `{pname}` ({ptype}{req}): {pdesc}')
                parts.append('')

        return '\n'.join(parts)

    def _format_oracle_response_text(self, decision: PlannerDecision) -> str:
        """Format oracle decision's content + tool calls for critic viewing."""
        parts = []
        if decision.response_content:
            parts.append(decision.response_content)
        for tc in decision.response_tool_calls:
            if isinstance(tc, dict):
                name = tc.get('name', '?')
                args = json.dumps(tc.get('arguments', {}))
                parts.append(f'[TOOL CALL] {name}({args})')
            else:
                # Planner LLM sometimes returns malformed tool calls as strings
                parts.append(f'[TOOL CALL] {tc}')
        return '\n'.join(parts)

    # ------------------------------------------------------------------
    # Synthetic ModelResponse construction
    # ------------------------------------------------------------------

    def _build_synthetic_response(self, decision: PlannerDecision) -> ModelResponse:
        """Build a ModelResponse from oracle planner's revise/rewrite output.

        This synthetic response replaces the blinded solver's response entirely.
        It gets recorded in tool_call_metadata.model_response on the resulting
        action, so the trajectory contains the oracle's version as the LLM
        completion — which is what we want for SFT data.
        """
        from litellm.types.utils import ChatCompletionMessageToolCall, Function

        tool_calls = []
        for i, tc in enumerate(decision.response_tool_calls):
            if not isinstance(tc, dict):
                logger.warning(
                    f'[OracleGuided] Skipping malformed tool call at index {i}: {tc!r}'
                )
                continue
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=f'oracle_{decision.step_index}_{i}_{uuid4().hex[:8]}',
                    type='function',
                    function=Function(
                        name=tc.get('name', ''),
                        arguments=json.dumps(tc.get('arguments', {})),
                    ),
                )
            )

        message = LitellmMessage(
            content=decision.response_content or '',
            role='assistant',
            tool_calls=tool_calls if tool_calls else None,
        )

        choice = Choices(
            finish_reason='tool_calls' if tool_calls else 'stop',
            index=0,
            message=message,
        )

        return ModelResponse(
            id=f'oracle-guided-{uuid4().hex}',
            choices=[choice],
            model='oracle-guided',
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
