"""Leakage Critic for the Oracle Guided V2 agent.

Validates oracle planner revisions using a combined neural judgment +
symbolic regex extraction approach.  Optionally rechecks failed regex
patterns with a focused LLM call.

This is the V2 version of HybridCritic, focused solely on information
leakage detection. The stage-transition and fact-relevance checking has
been removed (handled by SufficiencyCritic instead).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jinja2

from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM_RETRY_EXCEPTIONS

_CRITIC_TRANSIENT_RETRIES = 3
_CRITIC_RETRY_WAIT = 10  # seconds


def _llm_call_with_transient_retry(llm, **kwargs):
    """Call llm.completion with inner retries for transient network errors."""
    for attempt in range(_CRITIC_TRANSIENT_RETRIES):
        try:
            return llm.completion(**kwargs)
        except LLM_RETRY_EXCEPTIONS as exc:
            if attempt < _CRITIC_TRANSIENT_RETRIES - 1:
                logger.warning(
                    f'[LeakageCritic] Transient LLM error '
                    f'(attempt {attempt + 1}/{_CRITIC_TRANSIENT_RETRIES}), '
                    f'retrying in {_CRITIC_RETRY_WAIT}s: {exc}'
                )
                time.sleep(_CRITIC_RETRY_WAIT)
            else:
                raise


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LeakageCriticResult:
    step_index: int
    valid: bool
    neural_valid: bool
    neural_reasons: list[str] = field(default_factory=list)
    symbolic_checks: list[dict] = field(default_factory=list)
    symbolic_failures: list[dict] = field(default_factory=list)
    rechecked_failures: list[dict] = field(default_factory=list)
    feedback_message: str = ''
    raw_judge_response: str = ''
    raw_recheck_response: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'step_index': self.step_index,
            'valid': self.valid,
            'neural_valid': self.neural_valid,
            'neural_reasons': self.neural_reasons,
            'symbolic_checks': self.symbolic_checks,
            'symbolic_failures': self.symbolic_failures,
            'rechecked_failures': self.rechecked_failures,
            'feedback_message': self.feedback_message,
            'raw_judge_response': self.raw_judge_response,
            'raw_recheck_response': self.raw_recheck_response,
        }


# ---------------------------------------------------------------------------
# Leakage Critic
# ---------------------------------------------------------------------------

class LeakageCritic:
    """Neural + symbolic critic for detecting information leakage in oracle revisions."""

    def __init__(self, llm: Any, issue_text: str) -> None:
        self.llm = llm
        self.issue_text = issue_text

        self.max_json_parse_retries = int(
            os.environ.get('GUIDED_V2_LEAKAGE_CRITIC_JSON_PARSE_MAX_RETRIES', '3')
        )

        prompts_dir = Path(__file__).parent / 'prompts'
        self._jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(prompts_dir)),
            undefined=jinja2.StrictUndefined,
        )

        self._save_prompts_dir: str | None = os.environ.get(
            'GUIDED_V2_LEAKAGE_CRITIC_SAVE_PROMPTS_DIR'
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        step_index: int,
        best_candidate_text: str,
        oracle_response_text: str,
        history_text: str,
        attempt: int = 0,
    ) -> LeakageCriticResult:
        """Validate oracle planner's revised response for information leakage."""
        # Phase 1: Neural judgment + symbolic check extraction
        judge_prompt = self._render_judge_prompt(
            best_candidate_text, oracle_response_text, history_text
        )

        judge_data = None
        raw_judge = ''
        for retry in range(self.max_json_parse_retries + 1):
            try:
                response = _llm_call_with_transient_retry(
                    self.llm,
                    messages=[{'role': 'user', 'content': judge_prompt}],
                    response_format={'type': 'json_object'},
                )
                raw_judge = response.choices[0].message.content or ''
                self._save_prompt(step_index, attempt, retry, 'judge', judge_prompt, raw_judge)

                judge_data = self._extract_json(raw_judge)
                if judge_data is not None:
                    break
                logger.warning(
                    f'[LeakageCritic] Judge JSON parse retry {retry + 1}'
                )
            except Exception as exc:
                logger.warning(f'[LeakageCritic] Judge LLM error: {exc}')
                raw_judge = str(exc)
                self._save_prompt(step_index, attempt, retry, 'judge', judge_prompt, raw_judge)

        # Fail-open on parse failure
        if judge_data is None:
            logger.warning(
                '[LeakageCritic] All judge parse retries exhausted. Fail-open (valid=True).'
            )
            return LeakageCriticResult(
                step_index=step_index,
                valid=True,
                neural_valid=True,
                feedback_message='',
                raw_judge_response=raw_judge,
            )

        # Extract neural judgment
        neural = judge_data.get('neural_judgment', {})
        neural_valid = neural.get('valid', True)
        neural_reasons = neural.get('reasons', [])
        if not isinstance(neural_reasons, list):
            neural_reasons = [str(neural_reasons)] if neural_reasons else []

        if neural_reasons and neural_valid:
            neural_valid = False

        # Phase 1.5: Realism check — detect leaked fact IDs and oracle terms
        leak_reasons = self._check_realism(oracle_response_text)
        if leak_reasons:
            neural_valid = False
            neural_reasons.extend(leak_reasons)
            logger.info(
                f'[LeakageCritic] Realism check found {len(leak_reasons)} leak(s) '
                f'in oracle response.'
            )

        # Extract symbolic checks
        symbolic_checks = judge_data.get('symbolic_checks', [])
        if not isinstance(symbolic_checks, list):
            symbolic_checks = []

        # Phase 2: Run symbolic regex checks on history
        symbolic_failures, symbolic_log = self._run_symbolic_checks(symbolic_checks, history_text)

        # Append symbolic check results to saved prompt
        if symbolic_log:
            self._append_to_saved_prompt(
                step_index, attempt, 'judge',
                '\n\n' + '=' * 80 + '\n'
                '=== SYMBOLIC CHECK RESULTS ===\n'
                + '=' * 80 + '\n\n'
                + symbolic_log
            )

        # Phase 3: Recheck on disagreement (neural=valid but symbolic=fail)
        rechecked_failures: list[dict] = []
        raw_recheck = ''
        if symbolic_failures and neural_valid:
            recheck_prompt = self._render_recheck_prompt(
                symbolic_failures, oracle_response_text, history_text
            )
            for retry in range(self.max_json_parse_retries + 1):
                try:
                    response = _llm_call_with_transient_retry(
                        self.llm,
                        messages=[{'role': 'user', 'content': recheck_prompt}],
                        response_format={'type': 'json_object'},
                    )
                    raw_recheck = response.choices[0].message.content or ''
                    self._save_prompt(
                        step_index, attempt, retry, 'recheck',
                        recheck_prompt, raw_recheck
                    )

                    recheck_data = self._extract_json(raw_recheck)
                    if recheck_data is not None:
                        recheck_results = recheck_data.get('results', [])
                        for r in recheck_results:
                            if r.get('reject_valid', True):
                                rechecked_failures.append(r)
                        break
                    logger.warning(
                        f'[LeakageCritic] Recheck JSON parse retry {retry + 1}'
                    )
                except Exception as exc:
                    logger.warning(f'[LeakageCritic] Recheck LLM error: {exc}')
                    raw_recheck = str(exc)
                    self._save_prompt(
                        step_index, attempt, retry, 'recheck',
                        recheck_prompt, raw_recheck
                    )

        # Combine verdicts
        if not neural_valid and symbolic_failures:
            rechecked_failures = symbolic_failures
        symbolic_valid = len(rechecked_failures) == 0
        overall_valid = neural_valid and symbolic_valid

        # Build feedback if invalid
        feedback_message = ''
        if not overall_valid:
            feedback_message = self._build_feedback(
                neural_valid, neural_reasons,
                rechecked_failures, symbolic_failures,
                oracle_response_text,
            )
            reasons_summary = '; '.join(neural_reasons[:3]) if neural_reasons else 'symbolic failures'
            logger.info(
                f'[LeakageCritic] Rejected at step {step_index} '
                f'(neural={neural_valid}, symbolic={symbolic_valid}): '
                f'{reasons_summary[:200]}'
            )
        else:
            logger.info(
                f'[LeakageCritic] Passed at step {step_index} '
                f'(neural={neural_valid}, symbolic={symbolic_valid})'
            )

        return LeakageCriticResult(
            step_index=step_index,
            valid=overall_valid,
            neural_valid=neural_valid,
            neural_reasons=neural_reasons,
            symbolic_checks=symbolic_checks,
            symbolic_failures=symbolic_failures,
            rechecked_failures=rechecked_failures,
            feedback_message=feedback_message,
            raw_judge_response=raw_judge,
            raw_recheck_response=raw_recheck,
        )

    # ------------------------------------------------------------------
    # Realism check — detect leaked oracle/fact references
    # ------------------------------------------------------------------

    _REALISM_PATTERNS: list[tuple[str, 're.Pattern[str]']] = [
        (
            'Leaked fact ID reference (e.g., "f2 confirmed", "[f5]")',
            re.compile(
                r'(?:\[f\d{1,2}\]|'
                r'\bf\d{1,2}\b\s*(?:confirmed|unlocked|used|revealed))',
                re.IGNORECASE,
            ),
        ),
        (
            'Leaked artifact/node ID reference (e.g., "[edit1]", "[repro1]")',
            re.compile(
                r'\[(?:repro\d*|analysis\d*|plan\d*|edit\d+|val\d+)\]',
                re.IGNORECASE,
            ),
        ),
        (
            'Oracle/fact/node terminology leaked',
            re.compile(
                r'\b(?:oracle\s+(?:context|planner|decision)|'
                r'fact\s+(?:ID|id|graph|tracker)|'
                r'node\s+(?:ID|id)|'
                r'golden\s+patch|'
                r'unlocker\s+action|'
                r'phase\s+gat(?:e|ing)\s+(?:block|rule))\b',
                re.IGNORECASE,
            ),
        ),
        (
            'Next target fact reference',
            re.compile(
                r'\bnext[_\s]+target[_\s]*(?:fact|node)\b',
                re.IGNORECASE,
            ),
        ),
    ]

    @classmethod
    def _check_realism(cls, response_text: str) -> list[str]:
        """Scan response for leaked fact IDs and oracle terminology."""
        if not response_text:
            return []

        reasons: list[str] = []
        for description, pattern in cls._REALISM_PATTERNS:
            matches = pattern.findall(response_text)
            if matches:
                examples = [m.strip() for m in matches[:3] if m.strip()]
                if examples:
                    reasons.append(
                        f'[REALISM] {description}: '
                        f'found {", ".join(repr(e) for e in examples)}'
                    )
                else:
                    reasons.append(f'[REALISM] {description}')
        return reasons

    # ------------------------------------------------------------------
    # Symbolic check execution
    # ------------------------------------------------------------------

    def _run_symbolic_checks(
        self, checks: list[dict], history_text: str
    ) -> tuple[list[dict], str]:
        """Run regex patterns against history text. Return (failures, log_text)."""
        failures = []
        log_lines = []
        for i, check in enumerate(checks):
            pattern = check.get('regex', '')
            desc = check.get('description', '')
            if not pattern:
                continue
            try:
                match = re.search(pattern, history_text, re.IGNORECASE | re.DOTALL)
                if match:
                    snippet = match.group(0)
                    if len(snippet) > 120:
                        snippet = snippet[:120] + '...'
                    log_lines.append(
                        f'[CHECK {i + 1}] PASS | regex: `{pattern}`\n'
                        f'  desc: {desc}\n'
                        f'  matched: "{snippet}"'
                    )
                else:
                    log_lines.append(
                        f'[CHECK {i + 1}] FAIL | regex: `{pattern}`\n'
                        f'  desc: {desc}\n'
                        f'  matched: (nothing)'
                    )
                    failures.append({
                        'description': desc,
                        'regex': pattern,
                        'matched': False,
                    })
            except re.error as exc:
                log_lines.append(
                    f'[CHECK {i + 1}] ERROR | regex: `{pattern}`\n'
                    f'  desc: {desc}\n'
                    f'  error: {exc}'
                )
                logger.warning(
                    f'[LeakageCritic] Invalid regex "{pattern}": {exc}. Skipping check.'
                )
        log_text = '\n\n'.join(log_lines) if log_lines else ''
        return failures, log_text

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _render_judge_prompt(
        self, best_candidate: str, oracle_response: str, history_text: str,
    ) -> str:
        template = self._jinja_env.get_template('critic_judge.j2')
        return template.render(
            issue_text=self.issue_text,
            history_text=history_text,
            best_candidate_text=best_candidate,
            oracle_response_text=oracle_response,
        )

    def _render_recheck_prompt(
        self, failed_checks: list[dict], oracle_response: str, history_text: str,
    ) -> str:
        template = self._jinja_env.get_template('critic_recheck.j2')
        return template.render(
            failed_checks=failed_checks,
            oracle_response_text=oracle_response,
            history_text=history_text,
        )

    # ------------------------------------------------------------------
    # Feedback building
    # ------------------------------------------------------------------

    def _build_feedback(
        self,
        neural_valid: bool,
        neural_reasons: list[str],
        rechecked_failures: list[dict],
        symbolic_failures: list[dict],
        oracle_response: str,
    ) -> str:
        parts = ['[LEAKAGE CRITIC — ORACLE RESPONSE REJECTED]\n']

        if not neural_valid and neural_reasons:
            parts.append('## Neural Judgment Failures')
            for reason in neural_reasons:
                parts.append(f'  - {reason}')
            parts.append('')

        if rechecked_failures:
            parts.append('## Symbolic Check Failures (confirmed after recheck)')
            for fail in rechecked_failures:
                desc = fail.get('description', '')
                regex = fail.get('regex', '')
                reason = fail.get('reason', '')
                parts.append(f'  - {desc} (regex: `{regex}`)')
                if reason:
                    parts.append(f'    Recheck reason: {reason}')
            parts.append('')

        parts.append('## Rejected Response')
        parts.append(oracle_response)
        parts.append('')

        parts.append('## Requirements for Revision')
        parts.append('1. All knowledge in reasoning must be inferrable from the interaction history.')
        parts.append('2. All file paths, line numbers, and symbol names must be reachable from history context.')
        parts.append('3. No oracle/patch/test information leakage.')
        parts.append('4. Response must be logically incremental from history — no jumps.')

        return '\n'.join(parts)

    # ------------------------------------------------------------------
    # JSON extraction
    # ------------------------------------------------------------------

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
        stage: str, prompt: str, response: str,
    ) -> None:
        save_dir = self._save_prompts_dir
        if not save_dir:
            return
        os.makedirs(save_dir, exist_ok=True)
        suffix = f'_jsonretry_{retry:02d}' if retry > 0 else ''
        filename = f'step_{step_index:04d}_attempt_{attempt:02d}{suffix}_{stage}.txt'
        path = os.path.join(save_dir, filename)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(prompt)
                f.write('\n\n' + '=' * 80 + '\n')
                f.write(f'=== {stage.upper()} LLM RESPONSE ===\n')
                f.write('=' * 80 + '\n\n')
                f.write(response)
        except Exception as exc:
            logger.warning(f'[LeakageCritic] Failed to save prompt: {exc}')

    def _append_to_saved_prompt(
        self, step_index: int, attempt: int, stage: str, text: str,
    ) -> None:
        save_dir = self._save_prompts_dir
        if not save_dir:
            return
        filename = f'step_{step_index:04d}_attempt_{attempt:02d}_{stage}.txt'
        path = os.path.join(save_dir, filename)
        try:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(text)
        except Exception as exc:
            logger.warning(f'[LeakageCritic] Failed to append to prompt file: {exc}')

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        issue_text: str,
    ) -> 'LeakageCritic | None':
        """Create LeakageCritic from environment configuration."""
        from openhands.core.config.utils import get_llm_config_arg
        from openhands.llm.llm import LLM
        from openhands.llm.metrics import Metrics

        config_name = os.environ.get('GUIDED_V2_LEAKAGE_CRITIC_LLM_CONFIG', 'blinded_critic')
        config_file = os.environ.get('CONFIG_FILE', 'config.toml')
        llm_config = get_llm_config_arg(config_name, config_file)
        if llm_config is None:
            logger.warning(
                f'[LeakageCritic] No LLM config found for "{config_name}" in {config_file}. '
                'Leakage critic disabled.'
            )
            return None

        llm_config.log_completions = False
        metrics = Metrics(model_name=llm_config.model)
        try:
            llm = LLM(config=llm_config, service_id='leakage_critic', metrics=metrics)
        except Exception as exc:
            logger.warning(f'[LeakageCritic] Failed to init LLM: {exc}. Critic disabled.')
            return None

        logger.info(
            f'[LeakageCritic] Initialized with model={llm_config.model} config={config_name}'
        )
        return cls(llm=llm, issue_text=issue_text)
