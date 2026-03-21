from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from jinja2 import Environment, FileSystemLoader

from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM
from openhands.llm.metrics import Metrics


@dataclass
class PlannerDecision:
    step_index: int
    decision: str  # candidate | proposal
    best_candidate_index: int
    chosen_candidate_index: int | None
    reason: str
    proposal_response_text: str
    raw_planner_response: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'step_index': self.step_index,
            'decision': self.decision,
            'best_candidate_index': self.best_candidate_index,
            'chosen_candidate_index': self.chosen_candidate_index,
            'reason': self.reason,
            'proposal_response_text': self.proposal_response_text,
            'raw_planner_response': self.raw_planner_response,
        }


class OraclePlanner:
    """Oracle-aware planner that selects a debugger candidate or proposes guidance.

    The planner has access to the golden patch and golden test through
    ``oracle_context``. It must still produce leak-free guidance grounded in the
    current observed history.
    """

    PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'prompts')

    def __init__(self, llm: LLM, issue_text: str, oracle_context: str, tool_descriptions: str = '') -> None:
        self.llm = llm
        self.issue_text = issue_text
        self.oracle_context = oracle_context
        self.tool_descriptions = tool_descriptions
        self.max_json_parse_retries = max(
            int(os.environ.get('ORACLE_PLANNER_JSON_PARSE_MAX_RETRIES', '3')),
            0,
        )
        self._jinja_env = Environment(
            loader=FileSystemLoader(self.PROMPTS_DIR),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def plan(
        self,
        step_index: int,
        history_text: str,
        candidates: list[str],
        planner_feedback: str = '',
        attempt: int = 0,
    ) -> PlannerDecision:
        prompt = self._render_prompt(
            step_index=step_index,
            history_text=history_text,
            candidates=candidates,
            planner_feedback=planner_feedback,
        )

        for parse_retry in range(self.max_json_parse_retries + 1):
            try:
                response = self.llm.completion(messages=[{'role': 'user', 'content': prompt}])
                raw_text = response.choices[0].message.content or ''
            except Exception as exc:
                logger.warning(
                    f'[OraclePlanner] LLM call failed: {exc}. Falling back to best candidate 0.'
                )
                self._maybe_save_prompt(
                    step_index,
                    attempt,
                    parse_retry,
                    prompt,
                    f'[LLM CALL FAILED]: {exc}',
                )
                return PlannerDecision(
                    step_index=step_index,
                    decision='candidate',
                    best_candidate_index=0,
                    chosen_candidate_index=0,
                    reason=f'Planner LLM call failed: {exc}',
                    proposal_response_text='',
                    raw_planner_response='',
                )

            self._maybe_save_prompt(step_index, attempt, parse_retry, prompt, raw_text)
            parsed = self._parse_response_or_none(step_index, raw_text, len(candidates))
            if parsed is not None:
                return parsed

            if parse_retry < self.max_json_parse_retries:
                logger.warning(
                    '[OraclePlanner] JSON parse failed; retrying planner completion '
                    f'({parse_retry + 1}/{self.max_json_parse_retries}).'
                )

        logger.warning(
            '[OraclePlanner] JSON parse retries exhausted. Falling back to candidate 0.'
        )
        return PlannerDecision(
            step_index=step_index,
            decision='candidate',
            best_candidate_index=0,
            chosen_candidate_index=0,
            reason='Planner response was not parseable JSON after retries.',
            proposal_response_text='',
            raw_planner_response='',
        )

    def _render_prompt(
        self,
        step_index: int,
        history_text: str,
        candidates: list[str],
        planner_feedback: str,
    ) -> str:
        template = self._jinja_env.get_template('planner_select_or_propose.j2')
        return template.render(
            issue_text=self.issue_text,
            oracle_context=self.oracle_context,
            step_index=step_index,
            history_text=history_text,
            candidates=candidates,
            planner_feedback=planner_feedback,
            tool_descriptions=self.tool_descriptions,
        )

    @staticmethod
    def _extract_json(text: str) -> str | None:
        fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if fence:
            return fence.group(1)
        bare = re.search(r'\{.*\}', text, re.DOTALL)
        if bare:
            return bare.group(0)
        return None

    @classmethod
    def _parse_response_or_none(
        cls,
        step_index: int,
        raw_text: str,
        num_candidates: int,
    ) -> PlannerDecision | None:
        json_str = cls._extract_json(raw_text)
        if json_str is None:
            logger.warning('[OraclePlanner] Non-JSON response.')
            return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning(f'[OraclePlanner] Malformed JSON: {exc}.')
            return None

        best_idx = cls._safe_index(data.get('best_candidate_index'), num_candidates)
        decision = str(data.get('decision', 'candidate')).strip().lower()
        chosen_idx_raw = data.get('chosen_candidate_index')
        chosen_idx = cls._safe_index(chosen_idx_raw, num_candidates)
        reason = str(data.get('reason', ''))
        proposal_text = str(data.get('proposal_response', '') or '')

        if decision not in {'candidate', 'proposal'}:
            decision = 'candidate'

        if decision == 'candidate':
            if chosen_idx is None:
                chosen_idx = best_idx
            return PlannerDecision(
                step_index=step_index,
                decision='candidate',
                best_candidate_index=best_idx,
                chosen_candidate_index=chosen_idx,
                reason=reason or 'Planner selected a candidate.',
                proposal_response_text='',
                raw_planner_response=raw_text,
            )

        if not proposal_text.strip():
            logger.warning(
                '[OraclePlanner] Proposal decision without proposal text. Falling back to best candidate.'
            )
            return PlannerDecision(
                step_index=step_index,
                decision='candidate',
                best_candidate_index=best_idx,
                chosen_candidate_index=best_idx,
                reason=reason or 'Planner proposal empty; fallback to best candidate.',
                proposal_response_text='',
                raw_planner_response=raw_text,
            )

        return PlannerDecision(
            step_index=step_index,
            decision='proposal',
            best_candidate_index=best_idx,
            chosen_candidate_index=chosen_idx,
            reason=reason or 'Planner proposed a better next response.',
            proposal_response_text=proposal_text,
            raw_planner_response=raw_text,
        )

    @classmethod
    def _parse_response(
        cls,
        step_index: int,
        raw_text: str,
        num_candidates: int,
    ) -> PlannerDecision:
        parsed = cls._parse_response_or_none(step_index, raw_text, num_candidates)
        if parsed is not None:
            return parsed

        logger.warning('[OraclePlanner] Non-JSON response. Falling back to candidate 0.')
        return PlannerDecision(
            step_index=step_index,
            decision='candidate',
            best_candidate_index=0,
            chosen_candidate_index=0,
            reason='Planner response was not parseable JSON.',
            proposal_response_text='',
            raw_planner_response=raw_text,
        )

    @staticmethod
    def _safe_index(value: Any, num_candidates: int) -> int | None:
        if num_candidates <= 0:
            return 0
        try:
            index = int(value)
        except (TypeError, ValueError):
            return None
        if 0 <= index < num_candidates:
            return index
        return None

    @staticmethod
    def _maybe_save_prompt(
        step_index: int,
        attempt: int,
        parse_retry: int,
        prompt: str,
        raw_response: str,
    ) -> None:
        save_dir = os.environ.get('ORACLE_PLANNER_SAVE_PROMPTS_DIR', '')
        if not save_dir:
            return
        try:
            os.makedirs(save_dir, exist_ok=True)
            if parse_retry == 0:
                fname = f'step_{step_index:04d}_attempt_{attempt:02d}.txt'
            else:
                fname = (
                    f'step_{step_index:04d}_attempt_{attempt:02d}'
                    f'_jsonretry_{parse_retry:02d}.txt'
                )
            fpath = os.path.join(save_dir, fname)
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write('=== ORACLE PLANNER PROMPT ===\n')
                f.write(prompt)
                f.write('\n\n=== ORACLE PLANNER RAW RESPONSE ===\n')
                f.write(raw_response)
                f.write('\n')
        except Exception as exc:
            logger.warning(f'[OraclePlanner] Failed to save prompt debug file: {exc}')

    @classmethod
    def from_env(cls, issue_text: str, oracle_context: str, tool_descriptions: str = '') -> 'OraclePlanner | None':
        from openhands.core.config.utils import get_llm_config_arg

        config_name = os.environ.get('ORACLE_PLANNER_LLM_CONFIG', 'oracle_planner')
        config_file = os.environ.get('CONFIG_FILE', 'config.toml')
        llm_config = get_llm_config_arg(config_name, config_file)
        if llm_config is None:
            logger.warning(
                f'[OraclePlanner] No LLM config found for "{config_name}" in {config_file}. Planner disabled.'
            )
            return None

        llm_config.log_completions = False
        metrics = Metrics(model_name=llm_config.model)
        try:
            llm = LLM(config=llm_config, service_id='oracle_planner', metrics=metrics)
        except Exception as exc:
            logger.warning(f'[OraclePlanner] Failed to initialize planner LLM: {exc}. Planner disabled.')
            return None

        logger.info(
            f'[OraclePlanner] Initialized with model={llm_config.model} config_name={config_name}'
        )
        return cls(llm=llm, issue_text=issue_text, oracle_context=oracle_context, tool_descriptions=tool_descriptions)
