from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from jinja2 import Environment, FileSystemLoader

from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM
from openhands.llm.metrics import Metrics


@dataclass
class ProposalValidationResult:
    step_index: int
    valid: bool
    reason: str
    unjustified_knowledge: list[str] = field(default_factory=list)
    prerequisite_conditions: list[str] = field(default_factory=list)
    feedback_message: str = ''
    proposal_response_text: str = ''
    raw_critic_response: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'step_index': self.step_index,
            'valid': self.valid,
            'reason': self.reason,
            'unjustified_knowledge': self.unjustified_knowledge,
            'prerequisite_conditions': self.prerequisite_conditions,
            'feedback_message': self.feedback_message,
            'proposal_response_text': self.proposal_response_text,
            'raw_critic_response': self.raw_critic_response,
        }


class OracleProposalCritic:
    """Blinded critic for oracle-planner proposed responses."""

    PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'prompts')

    def __init__(self, llm: LLM, issue_text: str) -> None:
        self.llm = llm
        self.issue_text = issue_text
        self.max_json_parse_retries = max(
            int(os.environ.get('ORACLE_PROPOSAL_CRITIC_JSON_PARSE_MAX_RETRIES', '3')),
            0,
        )
        self._jinja_env = Environment(
            loader=FileSystemLoader(self.PROMPTS_DIR),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def validate(
        self,
        step_index: int,
        history_text: str,
        proposal_response_text: str,
        attempt: int = 0,
    ) -> ProposalValidationResult:
        prompt = self._render_prompt(step_index, history_text, proposal_response_text)

        for parse_retry in range(self.max_json_parse_retries + 1):
            try:
                response = self.llm.completion(messages=[{'role': 'user', 'content': prompt}])
                raw_text = response.choices[0].message.content or ''
            except Exception as exc:
                logger.warning(
                    f'[OracleProposalCritic] LLM call failed: {exc}. Allowing proposal by fail-open.'
                )
                self._maybe_save_prompt(
                    step_index,
                    attempt,
                    prompt,
                    f'[LLM CALL FAILED]: {exc}',
                    parse_retry,
                )
                return ProposalValidationResult(
                    step_index=step_index,
                    valid=True,
                    reason=f'Critic LLM call failed ({exc}); proposal allowed by default.',
                    proposal_response_text=proposal_response_text,
                    raw_critic_response='',
                )

            self._maybe_save_prompt(step_index, attempt, prompt, raw_text, parse_retry)
            result = self._parse_response_or_none(step_index, raw_text)
            if result is not None:
                result.proposal_response_text = proposal_response_text
                return result

            if parse_retry < self.max_json_parse_retries:
                logger.warning(
                    '[OracleProposalCritic] JSON parse failed; retrying critic completion '
                    f'({parse_retry + 1}/{self.max_json_parse_retries}).'
                )

        logger.warning(
            '[OracleProposalCritic] JSON parse retries exhausted. Allowing proposal by fail-open.'
        )
        return ProposalValidationResult(
            step_index=step_index,
            valid=True,
            reason='Critic response was not parseable JSON after retries; proposal allowed.',
            proposal_response_text=proposal_response_text,
            raw_critic_response='',
        )

    def _render_prompt(
        self, step_index: int, history_text: str, proposal_response_text: str
    ) -> str:
        template = self._jinja_env.get_template('validate_oracle_proposal.j2')
        return template.render(
            issue_text=self.issue_text,
            step_index=step_index,
            history_text=history_text,
            proposal_response_text=proposal_response_text,
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
        cls, step_index: int, raw_text: str
    ) -> ProposalValidationResult | None:
        json_str = cls._extract_json(raw_text)
        if json_str is None:
            logger.warning('[OracleProposalCritic] Non-JSON response.')
            return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning(f'[OracleProposalCritic] Malformed JSON: {exc}.')
            return None

        unjustified = [str(x) for x in data.get('unjustified_knowledge', [])]
        prereqs = [str(x) for x in data.get('prerequisite_conditions', [])]
        reason = str(data.get('reason', ''))

        if unjustified:
            valid = False
        else:
            valid = bool(data.get('valid', True))

        feedback_message = cls._build_feedback_message(valid, reason, unjustified, prereqs)
        return ProposalValidationResult(
            step_index=step_index,
            valid=valid,
            reason=reason,
            unjustified_knowledge=unjustified,
            prerequisite_conditions=prereqs,
            feedback_message=feedback_message,
            raw_critic_response=raw_text,
        )

    @classmethod
    def _parse_response(cls, step_index: int, raw_text: str) -> ProposalValidationResult:
        parsed = cls._parse_response_or_none(step_index, raw_text)
        if parsed is not None:
            return parsed
        return ProposalValidationResult(
            step_index=step_index,
            valid=True,
            reason='Critic response was not parseable JSON; proposal allowed.',
            raw_critic_response=raw_text,
        )

    @staticmethod
    def _build_feedback_message(
        valid: bool,
        reason: str,
        unjustified: list[str],
        prereqs: list[str],
    ) -> str:
        if valid:
            return ''

        lines: list[str] = [
            '[QA REVIEW - ORACLE PROPOSAL REJECTED]',
            '',
            f'Reason: {reason}',
        ]
        if unjustified:
            lines += ['', 'Unjustified knowledge detected:']
            lines += [f'  - {item}' for item in unjustified]
        if prereqs:
            lines += ['', 'Prerequisite conditions before this proposal is valid:']
            lines += [f'  - {item}' for item in prereqs]

        lines += [
            '',
            'Revise the proposal with these hard constraints:',
            '  1. Do not mention or imply the oracle/golden patch or test directly.',
            '  2. Keep all claims grounded in the observed interaction history only.',
            '  3. Use a strictly incremental next step with no logic jumps.',
            '  4. If a concrete fix detail was not observed in history, do not include it.',
        ]
        return '\n'.join(lines)

    @staticmethod
    def _maybe_save_prompt(
        step_index: int,
        attempt: int,
        prompt: str,
        raw_response: str,
        parse_retry: int = 0,
    ) -> None:
        save_dir = os.environ.get('ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS_DIR', '')
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
                f.write('=== ORACLE PROPOSAL CRITIC PROMPT ===\n')
                f.write(prompt)
                f.write('\n\n=== ORACLE PROPOSAL CRITIC RAW RESPONSE ===\n')
                f.write(raw_response)
                f.write('\n')
        except Exception as exc:
            logger.warning(f'[OracleProposalCritic] Failed to save prompt debug file: {exc}')

    @classmethod
    def from_env(cls, issue_text: str) -> 'OracleProposalCritic | None':
        from openhands.core.config.utils import get_llm_config_arg

        config_name = os.environ.get(
            'ORACLE_PROPOSAL_CRITIC_LLM_CONFIG',
            'blinded_critic',
        )
        config_file = os.environ.get('CONFIG_FILE', 'config.toml')
        llm_config = get_llm_config_arg(config_name, config_file)
        if llm_config is None:
            logger.warning(
                f'[OracleProposalCritic] No LLM config found for "{config_name}" in {config_file}. Critic disabled.'
            )
            return None

        llm_config.log_completions = False
        metrics = Metrics(model_name=llm_config.model)
        try:
            llm = LLM(config=llm_config, service_id='oracle_proposal_critic', metrics=metrics)
        except Exception as exc:
            logger.warning(f'[OracleProposalCritic] Failed to initialize critic LLM: {exc}. Critic disabled.')
            return None

        logger.info(
            f'[OracleProposalCritic] Initialized with model={llm_config.model} config_name={config_name}'
        )
        return cls(llm=llm, issue_text=issue_text)
