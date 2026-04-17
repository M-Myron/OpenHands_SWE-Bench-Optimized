"""Oracle Guided V2 CodeAct Agent.

Stageless multi-critic orchestration:
  Blinded Solver (candidates)
  → Oracle Planner (select / revise)
  → SufficiencyCritic (check if response is safe given unused facts)
  → LeakageCritic (check for information leakage, only on revise)

Key differences from V1:
- No stage transitions or stage gating
- SufficiencyCritic replaces stage-based action blocking
- LeakageCritic only fires on revise decisions (not select)
- Planner: select/revise only (no rewrite)
- Single combined retry budget
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
from openhands.agenthub.oracle_guided_v2_codeact_agent.fact_tracker_v2 import (
    FactTrackerV2,
)
from openhands.agenthub.oracle_guided_v2_codeact_agent.guided_config_v2 import GuidedConfigV2
from openhands.agenthub.oracle_guided_v2_codeact_agent.leakage_critic import (
    LeakageCritic,
    LeakageCriticResult,
)
from openhands.agenthub.oracle_guided_v2_codeact_agent.oracle_planner_v2 import (
    OraclePlannerV2,
    PlannerDecision,
)
from openhands.agenthub.oracle_guided_v2_codeact_agent.sufficiency_critic import (
    SufficiencyCritic,
    SufficiencyCriticResult,
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
# Per-process triage log
# ---------------------------------------------------------------------------

_TRIAGE_LOG: list[dict] = []
_TRIAGE_PID: int | None = None


def _ensure_triage_file() -> str:
    global _TRIAGE_PID
    pid = os.getpid()
    _TRIAGE_PID = pid
    return f'/tmp/oracle_guided_v2_{pid}.jsonl'


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

class OracleGuidedV2CodeActAgent(CodeActAgent):
    """Stageless oracle-guided agent with sufficiency + leakage critics."""

    VERSION = '1.0'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Load config
        self._guided_config = GuidedConfigV2.load()
        self._guided_config.export_to_env()

        # Component state — lazy init on first step
        self._oracle_planner: OraclePlannerV2 | None = None
        self._sufficiency_critic: SufficiencyCritic | None = None
        self._leakage_critic: LeakageCritic | None = None
        self._fact_tracker: FactTrackerV2 | None = None
        self._components_initialized: bool = False

        # Runtime config from env (after export_to_env)
        self._num_candidates = max(1, int(os.environ.get('GUIDED_V2_NUM_CANDIDATES', '1')))
        self._max_retries = max(0, int(os.environ.get('GUIDED_V2_MAX_RETRIES', '2')))
        self._oracle_start_step = max(0, int(os.environ.get('GUIDED_V2_ORACLE_START_STEP', '0')))
        self._oracle_auto_activate = os.environ.get('GUIDED_V2_ORACLE_AUTO_ACTIVATE', '0') == '1'
        self._oracle_auto_activate_fallback_step = max(
            1, int(os.environ.get('GUIDED_V2_ORACLE_AUTO_ACTIVATE_FALLBACK_STEP', '5'))
        )
        self._oracle_activated = False
        self._history_near_window = int(os.environ.get('GUIDED_V2_PLANNER_HISTORY_NEAR_WINDOW', '5'))

        self.guided_log: list[dict] = []

    # ------------------------------------------------------------------
    # Oracle auto-activation
    # ------------------------------------------------------------------

    _PHASE3_PATTERN = _re.compile(r'##\s*Phase\s*3', _re.IGNORECASE)

    def _should_oracle_be_active(
        self, step_index: int, history_events: list['Event'],
    ) -> bool:
        """Determine whether the oracle planner should be active this step."""
        if not self._oracle_auto_activate:
            return step_index >= self._oracle_start_step

        if self._oracle_activated:
            return True

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
                    f'[OracleGuidedV2] Auto-activation triggered: '
                    f'Phase 3 header detected at step {step_index}'
                )
                _append_triage({
                    'step_index': step_index,
                    'event': 'oracle_auto_activated',
                    'trigger': 'phase3_header',
                })
                return True

        if step_index >= self._oracle_auto_activate_fallback_step:
            self._oracle_activated = True
            logger.info(
                f'[OracleGuidedV2] Auto-activation triggered: '
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
            f'[OracleGuidedV2] Step {step_index}: oracle not yet active'
        )
        return False

    # ------------------------------------------------------------------
    # Sufficiency trigger
    # ------------------------------------------------------------------

    @staticmethod
    def _should_trigger_sufficiency_critic(response_text: str) -> str:
        """Determine if the SufficiencyCritic should evaluate this response.

        Returns a non-empty reason string (truthy) when triggered, or
        an empty string (falsy) when not triggered.

        Triggers on:
        - think tool call (AgentThinkAction)
        - str_replace_editor with 'create' or 'str_replace' command
        - Phase-transition keywords indicating move from exploration to
          analysis/implementation (e.g. "Phase 5", "FIX ANALYSIS")
        - Long reasoning text (>500 chars before tool call) suggesting
          heavy conclusions or planning that may be premature

        Does NOT trigger on pure exploration (bash grep/find, file view)
        since those are inherently safe actions — unless accompanied by
        the signals above.

        This is a separate function for easy modification.
        """
        # Check for think tool call
        if _re.search(r'"name"\s*:\s*"think"', response_text, _re.IGNORECASE):
            return 'think tool call'

        # Check for str_replace_editor with create or str_replace
        m = _re.search(
            r'str_replace_editor.*"command"\s*:\s*"(create|str_replace)"',
            response_text,
            _re.IGNORECASE | _re.DOTALL,
        )
        if m:
            return f'str_replace_editor {m.group(1)}'

        # Check for phase-transition keywords that signal moving beyond
        # exploration into analysis, planning, or implementation
        m = _re.search(
            r'(Phase\s*[5-8]|FIX\s+(?:ANALYSIS|IMPLEMENTATION)|'
            r'IMPLEMENTATION|VERIFICATION)',
            response_text,
        )
        if m:
            return f'phase keyword: {m.group(1)}'

        # Check for long reasoning text (>1500 chars before the first tool
        # call marker) — indicates heavy conclusions or planning
        tool_marker = _re.search(r'\[TOOL CALL\]', response_text)
        reasoning_text = response_text[:tool_marker.start()] if tool_marker else response_text
        reasoning_len = len(reasoning_text.strip())
        if reasoning_len > 1500:
            return f'long reasoning ({reasoning_len} chars)'

        return ''

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

        # Build messages for blinded solver
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

        _CANDIDATE_TRANSIENT_RETRIES = 5
        _CANDIDATE_RETRY_WAIT = 10

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
                    if retry_i < _CANDIDATE_TRANSIENT_RETRIES - 1:
                        logger.warning(
                            f'[OracleGuidedV2] Candidate {ci} transient error '
                            f'(attempt {retry_i + 1}/{_CANDIDATE_TRANSIENT_RETRIES}), '
                            f'retrying in {_CANDIDATE_RETRY_WAIT}s: {exc}'
                        )
                        time.sleep(_CANDIDATE_RETRY_WAIT)
                    else:
                        logger.warning(
                            f'[OracleGuidedV2] Candidate {ci} generation failed '
                            f'after {_CANDIDATE_TRANSIENT_RETRIES} transient retries: {exc}'
                        )
                        last_candidate_error = exc
                except Exception as exc:
                    # Treat any API/network error as retryable (e.g.
                    # litellm.APIError with 502/503 which isn't in
                    # LLM_RETRY_EXCEPTIONS).  Only give up on the last
                    # retry.
                    if retry_i < _CANDIDATE_TRANSIENT_RETRIES - 1:
                        logger.warning(
                            f'[OracleGuidedV2] Candidate {ci} error '
                            f'(attempt {retry_i + 1}/{_CANDIDATE_TRANSIENT_RETRIES}), '
                            f'retrying in {_CANDIDATE_RETRY_WAIT}s: '
                            f'{type(exc).__name__}: {str(exc)[:200]}'
                        )
                        time.sleep(_CANDIDATE_RETRY_WAIT)
                    else:
                        logger.warning(
                            f'[OracleGuidedV2] Candidate {ci} generation failed '
                            f'after {_CANDIDATE_TRANSIENT_RETRIES} retries: {exc}'
                        )
                        last_candidate_error = exc

        t_candidates = time.monotonic() - t0

        if not candidate_responses:
            if last_candidate_error is not None:
                logger.error(
                    f'[OracleGuidedV2] All {self._num_candidates} candidates '
                    f'failed. Re-raising last error.'
                )
                raise last_candidate_error
            logger.error('[OracleGuidedV2] All candidates failed. Returning finish action.')
            return AgentFinishAction()

        # =============================================================
        # Phase 2: Oracle Planner + Dual Critic Loop
        # =============================================================
        chosen_response: ModelResponse | None = None
        planner_feedback = ''
        t1 = time.monotonic()
        planner_times: list[float] = []
        sufficiency_times: list[float] = []
        leakage_times: list[float] = []
        final_attempt = 0
        final_decision_type = 'select'

        if self._oracle_planner is not None and self._should_oracle_be_active(step_index, state.history):
            history_text_planner = self._render_windowed_history(
                state.history, self._history_near_window,
                include_system_instruction=self._guided_config.planner.include_system_instruction,
            )
            history_text_critic = self._render_full_history(state.history)

            retries_used = 0

            for attempt in range(self._max_retries + 1):
                final_attempt = attempt

                # --- Planner decision ---
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
                    'event': 'planner_decision',
                    'attempt': attempt,
                    'decision': decision.decision,
                    'candidate_index': decision.candidate_index,
                    'facts_used': decision.facts_used,
                    'next_target_fact': decision.next_target_fact,
                    'reason': decision.reason,
                    'response_content': decision.response_content[:1000] if decision.response_content else '',
                })

                # --- Build the response text for checking ---
                if decision.decision == 'select':
                    response_text = candidate_texts[
                        min(decision.candidate_index, len(candidate_texts) - 1)
                    ]
                    tool_calls_text = ''
                else:
                    response_text = decision.response_content or ''
                    tool_calls_text = ' '.join(
                        json.dumps(tc) for tc in (decision.response_tool_calls or [])
                    )
                full_text = response_text + ' ' + tool_calls_text

                # --- SufficiencyCritic (if triggered) ---
                # For select decisions that are pure exploration (no file
                # creation / code edit / think), skip the LLM call — the
                # planner already judged the candidate productive and the
                # hard rule (instant, 0ms) catches premature file creation.
                sufficiency_ok = True
                _trigger_reason = ''
                if (
                    self._sufficiency_critic is not None
                    and self._fact_tracker is not None
                ):
                    _trigger_reason = self._should_trigger_sufficiency_critic(full_text)
                _trigger_sufficiency = bool(_trigger_reason)
                if _trigger_sufficiency:
                    logger.info(
                        f'[OracleGuidedV2] Sufficiency critic triggered at '
                        f'step {step_index}: {_trigger_reason}'
                    )
                    # Gather fact state
                    unused_data = self._fact_tracker.get_unused_facts_and_artifacts()
                    used_facts = self._fact_tracker.get_used_facts_summary()

                    t_suf_start = time.monotonic()
                    suf_result = self._sufficiency_critic.validate(
                        step_index=step_index,
                        response_text=response_text,
                        tool_calls_text=tool_calls_text,
                        used_facts=used_facts,
                        unused_facts=unused_data['unused_facts'],
                        unused_artifacts=unused_data['unused_artifacts'],
                        attempt=attempt,
                    )
                    sufficiency_times.append(round(time.monotonic() - t_suf_start, 2))

                    _append_triage({
                        'step_index': step_index,
                        'event': 'sufficiency_critic_result',
                        'attempt': attempt,
                        'trigger': _trigger_reason,
                        'passed': suf_result.passed,
                        'hard_rule_failed': suf_result.hard_rule_failed,
                        'reason': suf_result.reason,
                        'relevant_unused_facts': suf_result.relevant_unused_facts,
                        'mark_used': suf_result.mark_used,
                    })

                    # If the critic passed with mark_used, mark those
                    # artifacts as used in the fact tracker now.
                    if suf_result.passed and suf_result.mark_used and self._fact_tracker:
                        self._fact_tracker.mark_used(
                            suf_result.mark_used, step_index, force=True,
                        )
                        logger.info(
                            f'[OracleGuidedV2] Sufficiency critic marked '
                            f'artifacts as used: {suf_result.mark_used}'
                        )

                    if not suf_result.passed:
                        sufficiency_ok = False
                        retries_used += 1

                        planner_feedback = (
                            '[SUFFICIENCY CRITIC — RESPONSE REJECTED]\n\n'
                            + suf_result.feedback
                            + '\n\n## YOUR REJECTED RESPONSE (do NOT repeat this):\n'
                            + full_text[:1000]
                            + '\n\n## IMPORTANT: You MUST produce a '
                            'DIFFERENT TYPE of action (e.g., view a file, '
                            'grep for a symbol, run a test) — do NOT '
                            'produce another variant of the same rejected '
                            'action.'
                        )

                        if retries_used > self._max_retries:
                            # Exhausted — accept last response
                            logger.warning(
                                f'[OracleGuidedV2] Retries exhausted at step {step_index}. '
                                f'Accepting last response (sufficiency fail).'
                            )
                            sufficiency_ok = True  # force accept

                if not sufficiency_ok:
                    continue  # retry with feedback

                # --- LeakageCritic (only on revise) ---
                leakage_ok = True
                if decision.decision == 'revise' and self._leakage_critic is not None:
                    oracle_response_text = self._format_oracle_response_text(decision)

                    t_leak_start = time.monotonic()
                    leak_result = self._leakage_critic.validate(
                        step_index=step_index,
                        best_candidate_text=candidate_texts[0],
                        oracle_response_text=oracle_response_text,
                        history_text=history_text_critic,
                        attempt=attempt,
                    )
                    leakage_times.append(round(time.monotonic() - t_leak_start, 2))

                    _append_triage({
                        'step_index': step_index,
                        'event': 'leakage_critic_result',
                        'attempt': attempt,
                        'valid': leak_result.valid,
                        'neural_valid': leak_result.neural_valid,
                        'neural_reasons': leak_result.neural_reasons,
                        'symbolic_failures': len(leak_result.symbolic_failures),
                        'rechecked_failures': len(leak_result.rechecked_failures),
                    })

                    if not leak_result.valid:
                        leakage_ok = False
                        retries_used += 1

                        planner_feedback = leak_result.feedback_message

                        if retries_used > self._max_retries:
                            logger.warning(
                                f'[OracleGuidedV2] Retries exhausted at step {step_index}. '
                                f'Accepting last response (leakage fail).'
                            )
                            leakage_ok = True  # force accept

                if not leakage_ok:
                    continue  # retry with feedback

                # --- Accept ---
                if decision.decision == 'select':
                    idx = decision.candidate_index
                    if 0 <= idx < len(candidate_responses):
                        chosen_response = candidate_responses[idx]
                    else:
                        chosen_response = candidate_responses[0]
                else:
                    chosen_response = self._build_synthetic_response(decision)

                # Mark facts used
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
            self._fact_tracker.check_and_set_impl_complete(step_index)

        total_time = time.monotonic() - t0
        t_planner_total = sum(planner_times)
        t_sufficiency_total = sum(sufficiency_times)
        t_leakage_total = sum(leakage_times)
        _append_triage({
            'step_index': step_index,
            'event': 'step_timing',
            'candidates_sec': round(t_candidates, 2),
            'planner_sec': round(t_planner_total, 2),
            'sufficiency_critic_sec': round(t_sufficiency_total, 2),
            'leakage_critic_sec': round(t_leakage_total, 2),
            'planner_critic_sec': round(t_planner_critic, 2),
            'total_sec': round(total_time, 2),
            'num_candidates': len(candidate_responses),
            'attempts': final_attempt + 1,
            'decision': final_decision_type,
            'planner_times': planner_times,
            'sufficiency_times': sufficiency_times,
            'leakage_times': leakage_times,
        })

        # Build timing string
        timing_parts = [
            f'candidates={t_candidates:.1f}s ({len(candidate_responses)})',
        ]
        if len(planner_times) == 1 and not sufficiency_times and not leakage_times:
            timing_parts.append(f'planner={planner_times[0]:.1f}s ({final_decision_type})')
        else:
            for i, pt in enumerate(planner_times):
                timing_parts.append(f'plan[{i}]={pt:.1f}s')
            for i, st in enumerate(sufficiency_times):
                timing_parts.append(f'suf[{i}]={st:.1f}s')
            for i, lt in enumerate(leakage_times):
                timing_parts.append(f'leak[{i}]={lt:.1f}s')
            timing_parts.append(f'decision={final_decision_type}')
            if final_attempt > 0:
                timing_parts.append(f'attempts={final_attempt + 1}')

        timing_parts.append(f'total={total_time:.1f}s')
        logger.info(f'[OracleGuidedV2] Step {step_index}: {", ".join(timing_parts)}')

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
        """Lazy-initialize oracle planner, critics, and fact tracker."""
        self._components_initialized = True

        # Load oracle context
        oracle_context_text, react_facts_data = self._load_oracle_context()
        if not oracle_context_text and react_facts_data is None:
            logger.warning('[OracleGuidedV2] No oracle context found. Running as plain CodeActAgent.')
            return

        # Extract issue text and tool instructions from history
        issue_text = ''
        tool_instructions = ''
        for event in state.history:
            if isinstance(event, MessageAction) and event.source == 'user':
                issue_text = self._extract_issue_description(event.content)
                break

        for event in state.history:
            if hasattr(event, 'content') and hasattr(event, 'source'):
                if getattr(event, 'source', '') == 'system' or (
                    hasattr(event, '__class__') and 'SystemMessage' in event.__class__.__name__
                ):
                    tool_instructions = self._extract_tool_instructions(event.content)
                    break

        if not tool_instructions:
            tool_instructions = self._build_tool_descriptions()

        # Initialize fact tracker
        if react_facts_data:
            self._fact_tracker = FactTrackerV2(react_facts_data)
            self._fact_tracker.set_finish_extension_budget(
                self._guided_config.agent.finish_extension_steps
            )
            logger.info(
                f'[OracleGuidedV2] FactTrackerV2 initialized with '
                f'{len(self._fact_tracker.node_ids)} nodes.'
            )

        # Initialize oracle planner
        self._oracle_planner = OraclePlannerV2.from_env(
            issue_text=issue_text,
            oracle_context=oracle_context_text,
            tool_instructions=tool_instructions,
            fact_tracker=self._fact_tracker,
        )
        if self._oracle_planner:
            logger.info('[OracleGuidedV2] Oracle planner V2 initialized.')

        # Initialize sufficiency critic
        self._sufficiency_critic = SufficiencyCritic.from_env(
            issue_text=issue_text,
        )
        if self._sufficiency_critic:
            logger.info('[OracleGuidedV2] Sufficiency critic initialized.')

        # Initialize leakage critic
        self._leakage_critic = LeakageCritic.from_env(
            issue_text=issue_text,
        )
        if self._leakage_critic:
            logger.info('[OracleGuidedV2] Leakage critic initialized.')

    # ------------------------------------------------------------------
    # Oracle context loading
    # ------------------------------------------------------------------

    def _load_oracle_context(self) -> tuple[str, dict | None]:
        """Load oracle context from JSON file."""
        context_path = os.environ.get('ORACLE_GUIDED_CONTEXT_PATH', '').strip()
        if not context_path or not os.path.isfile(context_path):
            return '', None

        try:
            with open(context_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f'[OracleGuidedV2] Failed to load oracle context: {exc}')
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
        logger.info(f'[OracleGuidedV2] Oracle context loaded ({len(context_text)} chars).')
        return context_text, react_facts_data

    # ------------------------------------------------------------------
    # History rendering
    # ------------------------------------------------------------------

    def _render_windowed_history(
        self, events: list[Event], near_window: int,
        include_system_instruction: bool = True,
    ) -> str:
        """Render history with far events (action only) and near events (action + observation)."""
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

        if near_window <= 0 or near_window >= len(steps):
            cutoff = 0
        else:
            cutoff = len(steps) - near_window

        parts: list[str] = []

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
                parts.append(f'Step {i}: [ACTION] {step["action"]}')
            else:
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
        """Extract readable text from a ModelResponse."""
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
        match = re.search(
            r'<issue_description>(.*?)</issue_description>', text, re.DOTALL
        )
        if match:
            return match.group(1).strip()
        return text[:5000]

    def _extract_tool_instructions(self, system_message: str) -> str:
        """Extract tool usage instructions from system message."""
        if not system_message:
            return ''
        patterns = [
            r'(## Tools.*?)(?=\n## |\Z)',
            r'(# Tools.*?)(?=\n# |\Z)',
            r'(You have access to.*?tools.*?)(?=\n## |\n# |\Z)',
        ]
        for pat in patterns:
            match = re.search(pat, system_message, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
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
                parts.append(f'[TOOL CALL] {tc}')
        return '\n'.join(parts)

    # ------------------------------------------------------------------
    # Synthetic ModelResponse construction
    # ------------------------------------------------------------------

    def _build_synthetic_response(self, decision: PlannerDecision) -> ModelResponse:
        """Build a ModelResponse from oracle planner's revise output."""
        from litellm.types.utils import ChatCompletionMessageToolCall, Function

        tool_calls = []
        for i, tc in enumerate(decision.response_tool_calls):
            if not isinstance(tc, dict):
                logger.warning(
                    f'[OracleGuidedV2] Skipping malformed tool call at index {i}: {tc!r}'
                )
                continue
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=f'oracle_v2_{decision.step_index}_{i}_{uuid4().hex[:8]}',
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
            id=f'oracle-guided-v2-{uuid4().hex}',
            choices=[choice],
            model='oracle-guided-v2',
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
