"""Oracle Planner V2 for the Oracle Guided V2 agent.

Selects or revises blinded-solver candidates using private oracle context
and a structured investigation fact graph. Unlike V1, there is no stage
logic and no rewrite decision type.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jinja2

from openhands.agenthub.oracle_guided_v2_codeact_agent.fact_tracker_v2 import (
    FactTrackerV2,
)
from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM_RETRY_EXCEPTIONS

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PlannerDecision:
    step_index: int
    decision: str  # 'select' | 'revise'
    candidate_index: int
    response_content: str  # reasoning text (for revise)
    response_tool_calls: list[dict] = field(default_factory=list)
    facts_used: list[str] = field(default_factory=list)
    next_target_fact: str = ''
    reason: str = ''
    raw_planner_response: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'step_index': self.step_index,
            'decision': self.decision,
            'candidate_index': self.candidate_index,
            'response_content': self.response_content,
            'response_tool_calls': self.response_tool_calls,
            'facts_used': self.facts_used,
            'next_target_fact': self.next_target_fact,
            'reason': self.reason,
            'raw_planner_response': self.raw_planner_response,
        }


# ---------------------------------------------------------------------------
# Oracle Planner V2
# ---------------------------------------------------------------------------

class OraclePlannerV2:
    """Stageless oracle planner — select/revise only."""

    def __init__(
        self,
        llm: Any,
        issue_text: str,
        oracle_context: str,
        tool_instructions: str,
        fact_tracker: FactTrackerV2 | None,
    ) -> None:
        self.llm = llm
        self.issue_text = issue_text
        self.oracle_context = oracle_context
        self.tool_instructions = tool_instructions
        self.fact_tracker = fact_tracker

        self.max_json_parse_retries = int(
            os.environ.get('GUIDED_V2_PLANNER_JSON_PARSE_MAX_RETRIES', '3')
        )

        prompts_dir = Path(__file__).parent / 'prompts'
        self._jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(prompts_dir)),
            undefined=jinja2.StrictUndefined,
        )

        self._save_prompts_dir: str | None = os.environ.get(
            'GUIDED_V2_PLANNER_SAVE_PROMPTS_DIR'
        )

        # Decision history for cross-step continuity
        self._decision_history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        step_index: int,
        history_text: str,
        candidates: list[str],
        feedback: str = '',
        attempt: int = 0,
    ) -> PlannerDecision:
        """Generate a planning decision for the current step."""
        prompt = self._render_prompt(
            step_index=step_index,
            history_text=history_text,
            candidate_texts=candidates,
            feedback=feedback,
        )

        decision: PlannerDecision | None = None
        raw_response = ''

        import time as _time
        _PLANNER_TRANSIENT_RETRIES = 3
        _PLANNER_RETRY_WAIT = 10

        for retry in range(self.max_json_parse_retries + 1):
            try:
                response = None
                for net_retry in range(_PLANNER_TRANSIENT_RETRIES):
                    try:
                        response = self.llm.completion(
                            messages=[{'role': 'user', 'content': prompt}],
                            response_format={'type': 'json_object'},
                        )
                        break
                    except LLM_RETRY_EXCEPTIONS as net_exc:
                        if net_retry < _PLANNER_TRANSIENT_RETRIES - 1:
                            logger.warning(
                                f'[OraclePlannerV2] Transient LLM error '
                                f'(attempt {net_retry + 1}/{_PLANNER_TRANSIENT_RETRIES}), '
                                f'retrying in {_PLANNER_RETRY_WAIT}s: {net_exc}'
                            )
                            _time.sleep(_PLANNER_RETRY_WAIT)
                        else:
                            raise

                raw_response = response.choices[0].message.content or ''
                self._save_prompt(step_index, attempt, retry, prompt, raw_response)

                decision = self._parse_response(raw_response, step_index, len(candidates))
                if decision is not None:
                    break

                logger.warning(
                    f'[OraclePlannerV2] JSON parse retry {retry + 1}/{self.max_json_parse_retries + 1}'
                )
            except Exception as exc:
                logger.warning(f'[OraclePlannerV2] LLM call error: {exc}')
                raw_response = str(exc)
                self._save_prompt(step_index, attempt, retry, prompt, raw_response)

        if decision is None:
            logger.warning(
                f'[OraclePlannerV2] All parse retries exhausted at step {step_index}. '
                'Falling back to candidate 0.'
            )
            decision = PlannerDecision(
                step_index=step_index,
                decision='select',
                candidate_index=0,
                response_content='',
                reason='Fallback: JSON parse failed',
                raw_planner_response=raw_response,
            )

        return decision

    def record_accepted_decision(self, decision: PlannerDecision) -> None:
        """Record a decision that was accepted."""
        self._decision_history.append({
            'step': decision.step_index,
            'decision': decision.decision,
            'facts_used': decision.facts_used,
            'next_target': decision.next_target_fact,
            'reason': decision.reason,
        })

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _render_prompt(
        self,
        step_index: int,
        history_text: str,
        candidate_texts: list[str],
        feedback: str = '',
    ) -> str:
        template = self._jinja_env.get_template('planner_v2.j2')

        has_facts = self.fact_tracker is not None and self.fact_tracker.has_facts
        available_facts_text = ''
        usage_state_text = ''
        used_facts_summary = ''
        if has_facts:
            available_facts_text = self.fact_tracker.render_categorized_facts(
                history_text,
            )
            usage_state_text = self.fact_tracker.render_usage_state_for_planner()
            used_facts_summary = self.fact_tracker.render_used_facts_summary()

        # Build decision history text (last 8 steps)
        recent_decisions = self._decision_history[-8:]
        decision_history_text = ''
        if recent_decisions:
            lines = []
            for d in recent_decisions:
                facts_str = ', '.join(d['facts_used']) if d['facts_used'] else 'none'
                lines.append(
                    f'- Step {d["step"]}: **{d["decision"]}** | '
                    f'facts_used=[{facts_str}] | '
                    f'next_target={d["next_target"] or "none"}'
                )
            decision_history_text = '\n'.join(lines)

        return template.render(
            issue_text=self.issue_text,
            oracle_context=self.oracle_context,
            tool_instructions=self.tool_instructions,
            step_index=step_index,
            history_text=history_text,
            candidate_texts=candidate_texts,
            num_candidates=len(candidate_texts),
            has_facts=has_facts,
            available_facts_text=available_facts_text,
            usage_state_text=usage_state_text,
            feedback=feedback,
            decision_history_text=decision_history_text,
            used_facts_summary=used_facts_summary,
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self, raw: str, step_index: int, num_candidates: int,
    ) -> PlannerDecision | None:
        """Parse planner JSON response into a PlannerDecision."""
        try:
            data = self._extract_json(raw)
            if data is None:
                return None

            decision_type = data.get('decision', 'select')
            # V2: only select/revise — coerce rewrite to revise
            if decision_type == 'rewrite':
                decision_type = 'revise'
            if decision_type not in ('select', 'revise'):
                logger.warning(
                    f'[OraclePlannerV2] Invalid decision type: {decision_type}. '
                    'Coercing to select.'
                )
                decision_type = 'select'

            candidate_index = data.get('candidate_index', 0)
            if not isinstance(candidate_index, int) or candidate_index < 0:
                candidate_index = 0
            if candidate_index >= num_candidates:
                candidate_index = 0

            response_content = data.get('response_content', '')
            reason = data.get('reason', '')
            facts_used = data.get('facts_used', [])
            next_target_fact = data.get('next_target_fact', '')

            # Parse tool calls
            raw_tool_calls = data.get('response_tool_calls', data.get('response_tool_call'))
            tool_calls: list[dict] = []
            if raw_tool_calls:
                if isinstance(raw_tool_calls, dict):
                    tool_calls = [raw_tool_calls]
                elif isinstance(raw_tool_calls, list):
                    tool_calls = raw_tool_calls

            # Validate: revise must have content or tool_calls
            if decision_type == 'revise':
                if not response_content and not tool_calls:
                    logger.warning(
                        '[OraclePlannerV2] Revise decision has no content or '
                        'tool calls. Falling back to select.'
                    )
                    decision_type = 'select'

            return PlannerDecision(
                step_index=step_index,
                decision=decision_type,
                candidate_index=candidate_index,
                response_content=response_content,
                response_tool_calls=tool_calls,
                facts_used=facts_used if isinstance(facts_used, list) else [],
                next_target_fact=next_target_fact if isinstance(next_target_fact, str) else '',
                reason=reason,
                raw_planner_response=raw,
            )
        except Exception as exc:
            logger.warning(f'[OraclePlannerV2] Parse error: {exc}')
            return None

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Extract JSON object from text."""
        match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        text = text.strip()
        if text.startswith('{'):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        brace_start = text.find('{')
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start:i + 1])
                        except json.JSONDecodeError:
                            break
        return None

    # ------------------------------------------------------------------
    # Prompt saving
    # ------------------------------------------------------------------

    def _save_prompt(
        self, step_index: int, attempt: int, retry: int,
        prompt: str, response: str,
    ) -> None:
        save_dir = self._save_prompts_dir
        if not save_dir:
            return
        os.makedirs(save_dir, exist_ok=True)
        suffix = f'_jsonretry_{retry:02d}' if retry > 0 else ''
        filename = f'step_{step_index:04d}_attempt_{attempt:02d}{suffix}_plan.txt'
        path = os.path.join(save_dir, filename)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(prompt)
                f.write('\n\n' + '=' * 80 + '\n')
                f.write('=== LLM RESPONSE ===\n')
                f.write('=' * 80 + '\n\n')
                f.write(response)
        except Exception as exc:
            logger.warning(f'[OraclePlannerV2] Failed to save prompt: {exc}')

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        issue_text: str,
        oracle_context: str,
        tool_instructions: str,
        fact_tracker: FactTrackerV2 | None,
    ) -> 'OraclePlannerV2 | None':
        """Create OraclePlannerV2 from environment configuration."""
        from openhands.core.config.utils import get_llm_config_arg
        from openhands.llm.llm import LLM
        from openhands.llm.metrics import Metrics

        config_name = os.environ.get('GUIDED_V2_PLANNER_LLM_CONFIG', 'oracle_planner')
        config_file = os.environ.get('CONFIG_FILE', 'config.toml')
        llm_config = get_llm_config_arg(config_name, config_file)
        if llm_config is None:
            logger.warning(
                f'[OraclePlannerV2] No LLM config found for "{config_name}" in {config_file}. '
                'Planner disabled.'
            )
            return None

        llm_config.log_completions = False
        metrics = Metrics(model_name=llm_config.model)
        try:
            llm = LLM(config=llm_config, service_id='oracle_planner_v2', metrics=metrics)
        except Exception as exc:
            logger.warning(f'[OraclePlannerV2] Failed to init LLM: {exc}. Planner disabled.')
            return None

        logger.info(
            f'[OraclePlannerV2] Initialized with model={llm_config.model} config={config_name}'
        )
        return cls(
            llm=llm,
            issue_text=issue_text,
            oracle_context=oracle_context,
            tool_instructions=tool_instructions,
            fact_tracker=fact_tracker,
        )
