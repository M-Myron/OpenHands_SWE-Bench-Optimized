"""Hybrid Critic for the Oracle Guided agent.

Validates oracle planner revisions using a combined neural judgment +
symbolic regex extraction approach.  Optionally rechecks failed regex
patterns with a focused LLM call.
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

_CRITIC_TRANSIENT_RETRIES = int(os.environ.get('GUIDED_TRANSIENT_RETRIES', '5'))
_CRITIC_RETRY_BASE_WAIT = int(os.environ.get('GUIDED_RETRY_BASE_WAIT', '10'))


def _llm_call_with_transient_retry(llm, **kwargs):
    """Call llm.completion with inner retries for transient network errors."""
    for attempt in range(_CRITIC_TRANSIENT_RETRIES):
        try:
            return llm.completion(**kwargs)
        except LLM_RETRY_EXCEPTIONS as exc:
            if attempt < _CRITIC_TRANSIENT_RETRIES - 1:
                wait = _CRITIC_RETRY_BASE_WAIT * (2 ** attempt)
                logger.warning(
                    f'[HybridCritic] Transient LLM error '
                    f'(attempt {attempt + 1}/{_CRITIC_TRANSIENT_RETRIES}), '
                    f'retrying in {wait}s: {exc}'
                )
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CriticResult:
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
# Hybrid Critic
# ---------------------------------------------------------------------------

class HybridCritic:
    """Neural + symbolic critic for validating oracle planner revisions."""

    def __init__(self, llm: Any, issue_text: str) -> None:
        self.llm = llm
        self.issue_text = issue_text

        self.max_json_parse_retries = int(
            os.environ.get('GUIDED_CRITIC_JSON_PARSE_MAX_RETRIES', '3')
        )
        # When disabled, the judge prompt omits the symbolic-check section,
        # no regex execution happens, and the recheck LLM call is skipped.
        # Neural judgment + realism checks still run.
        self.enable_symbolic_checks = (
            os.environ.get('GUIDED_CRITIC_ENABLE_SYMBOLIC_CHECKS', '1').strip()
            not in ('0', 'false', 'False', '')
        )

        prompts_dir = Path(__file__).parent / 'prompts'
        self._jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(prompts_dir)),
            undefined=jinja2.StrictUndefined,
        )

        self._save_prompts_dir: str | None = os.environ.get(
            'GUIDED_CRITIC_SAVE_PROMPTS_DIR'
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_facts_relevance(
        self,
        unused_facts: list[dict],
        step_index: int = 0,
        attempt: int = 0,
    ) -> dict:
        """Ask the critic whether unused investigation facts are relevant
        to solving the current issue.

        Args:
            unused_facts: list of ``{'id': ..., 'statement': ...}`` dicts
                describing facts the solver has not explored.
            step_index: current step index (for prompt saving).
            attempt: current gate attempt (for prompt saving).

        Returns:
            dict with keys:
            - ``all_irrelevant`` (bool): True if ALL facts are irrelevant.
            - ``relevant_ids`` (list[str]): IDs of facts deemed relevant.
            - ``irrelevant_ids`` (list[str]): IDs of facts deemed irrelevant.
            - ``reason`` (str): brief explanation from the critic.
        """
        default_result = {
            'all_irrelevant': False,
            'relevant_ids': [f['id'] for f in unused_facts],
            'irrelevant_ids': [],
            'reason': '',
        }
        if not unused_facts:
            return {'all_irrelevant': True, 'relevant_ids': [],
                    'irrelevant_ids': [], 'reason': 'no facts to check'}

        fact_lines = '\n'.join(
            f'- [{f["id"]}] {f["statement"]}' for f in unused_facts
        )
        prompt = (
            'You are evaluating whether certain investigation areas are '
            'relevant to solving a software issue.\n\n'
            '# Issue Description\n\n'
            f'{self.issue_text}\n\n'
            '# Unexplored Investigation Areas\n\n'
            f'{fact_lines}\n\n'
            '# Task\n\n'
            'The solver has already explored the main code paths related to '
            'the issue and wants to proceed to creating a reproduction script. '
            'However, these investigation areas have not been explored yet.\n\n'
            'For EACH investigation area, determine whether it is **relevant** '
            'or **irrelevant** to the reported issue.\n\n'
            'Return a single JSON object:\n'
            '```json\n'
            '{"verdict": "all_irrelevant" | "some_relevant",\n'
            ' "relevant": ["id1", "id2"],\n'
            ' "irrelevant": ["id3", "id4"],\n'
            ' "reason": "brief explanation"}\n'
            '```'
        )

        try:
            response = _llm_call_with_transient_retry(
                self.llm,
                messages=[{'role': 'user', 'content': prompt}],
                response_format={'type': 'json_object'},
            )
            raw = response.choices[0].message.content or ''

            # Save prompt + response
            self._save_prompt(
                step_index, attempt, 0,
                'fact_relevance', prompt, raw,
            )

            data = json.loads(raw)
            verdict = data.get('verdict', 'some_relevant').lower().strip()
            reason = data.get('reason', '')

            # Parse results — prefer list format, fall back to per_fact
            relevant_ids: list[str] = []
            irrelevant_ids: list[str] = []
            fact_id_set = {f['id'] for f in unused_facts}

            raw_relevant = data.get('relevant', [])
            raw_irrelevant = data.get('irrelevant', [])
            per_fact = data.get('per_fact', [])

            if raw_relevant or raw_irrelevant:
                # New list format: {"relevant": [...], "irrelevant": [...]}
                for fid in raw_relevant:
                    if fid in fact_id_set:
                        relevant_ids.append(fid)
                for fid in raw_irrelevant:
                    if fid in fact_id_set:
                        irrelevant_ids.append(fid)
                # Any IDs not covered default to relevant
                for f in unused_facts:
                    if f['id'] not in relevant_ids and f['id'] not in irrelevant_ids:
                        relevant_ids.append(f['id'])
            elif per_fact and isinstance(per_fact, list):
                # Legacy per_fact format (backwards compat)
                for pf in per_fact:
                    fid = pf.get('id', '')
                    if fid in fact_id_set:
                        if pf.get('relevant', True):
                            relevant_ids.append(fid)
                        else:
                            irrelevant_ids.append(fid)
                for f in unused_facts:
                    if f['id'] not in relevant_ids and f['id'] not in irrelevant_ids:
                        relevant_ids.append(f['id'])
            else:
                # No breakdown — use verdict
                if verdict == 'all_irrelevant':
                    irrelevant_ids = [f['id'] for f in unused_facts]
                else:
                    relevant_ids = [f['id'] for f in unused_facts]

            all_irrelevant = len(relevant_ids) == 0
            logger.info(
                f'[HybridCritic] Facts relevance check: verdict={verdict}, '
                f'relevant={relevant_ids}, irrelevant={irrelevant_ids}, '
                f'reason={reason[:120]}'
            )
            return {
                'all_irrelevant': all_irrelevant,
                'relevant_ids': relevant_ids,
                'irrelevant_ids': irrelevant_ids,
                'reason': reason,
            }
        except Exception as exc:
            logger.warning(
                f'[HybridCritic] Facts relevance check failed: {exc}. '
                'Assuming facts are relevant (safe default).'
            )
            return default_result

    def validate(
        self,
        step_index: int,
        best_candidate_text: str,
        oracle_response_text: str,
        history_text: str,
        attempt: int = 0,
    ) -> CriticResult:
        """Validate oracle planner's revised/rewritten response."""
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
                    f'[HybridCritic] Judge JSON parse retry {retry + 1}'
                )
            except Exception as exc:
                logger.warning(f'[HybridCritic] Judge LLM error: {exc}')
                raw_judge = str(exc)
                self._save_prompt(step_index, attempt, retry, 'judge', judge_prompt, raw_judge)

        # Fail-open on parse failure
        if judge_data is None:
            logger.warning(
                '[HybridCritic] All judge parse retries exhausted. Fail-open (valid=True).'
            )
            return CriticResult(
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

        # Force invalid if neural says invalid
        if neural_reasons and neural_valid:
            neural_valid = False

        # Phase 1.5: Realism check — detect leaked fact IDs and oracle terms
        # in the oracle_response_text.  This is a programmatic check (no LLM
        # call) that catches leaks the sanitizer might miss, e.g. in tool-call
        # arguments or reasoning text.
        leak_reasons = self._check_realism(oracle_response_text)
        if leak_reasons:
            neural_valid = False
            neural_reasons.extend(leak_reasons)
            logger.info(
                f'[HybridCritic] Realism check found {len(leak_reasons)} leak(s) '
                f'in oracle response.'
            )

        # Extract symbolic checks (only when enabled — template omits the
        # section otherwise, so the LLM shouldn't produce them).
        if self.enable_symbolic_checks:
            symbolic_checks = judge_data.get('symbolic_checks', [])
            if not isinstance(symbolic_checks, list):
                symbolic_checks = []
        else:
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

        # Phase 3: Recheck failed regexes — ONLY when neural says valid but
        # symbolic disagrees (the disagreement case needing a tiebreaker).
        # If neural already says invalid, no need to recheck — just reject.
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
                        f'[HybridCritic] Recheck JSON parse retry {retry + 1}'
                    )
                except Exception as exc:
                    logger.warning(f'[HybridCritic] Recheck LLM error: {exc}')
                    raw_recheck = str(exc)
                    self._save_prompt(
                        step_index, attempt, retry, 'recheck',
                        recheck_prompt, raw_recheck
                    )

        # Combine verdicts
        # When neural=invalid, symbolic failures are auto-confirmed (no recheck needed).
        # When neural=valid, only rechecked failures count (recheck may dismiss false positives).
        if not neural_valid and symbolic_failures:
            # Neural already rejected — treat all symbolic failures as confirmed
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

        return CriticResult(
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

    # Patterns that should never appear in SFT training data
    _REALISM_PATTERNS: list[tuple[str, 're.Pattern[str]']] = [
        (
            'Leaked fact ID reference (e.g., "f2 confirmed", "[f5]")',
            re.compile(
                # Match fact IDs like f1, f2, ..., f99 — these never appear
                # naturally in developer text.  Require brackets or
                # confirmation words to avoid false positives on things like
                # "f-string" or variable names.
                r'(?:\[f\d{1,2}\]|'
                r'\bf\d{1,2}\b\s*(?:confirmed|unlocked|used|revealed))',
                re.IGNORECASE,
            ),
        ),
        (
            'Leaked artifact/node ID reference (e.g., "[edit1]", "[repro1]")',
            re.compile(
                # Artifact node IDs only leak when written in bracket
                # notation — bare words like "plan" and "analysis" are
                # normal English and must not trigger false positives.
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
        """Scan response for leaked fact IDs and oracle terminology.

        Returns a list of rejection reasons (empty if clean).
        """
        if not response_text:
            return []

        reasons: list[str] = []
        for description, pattern in cls._REALISM_PATTERNS:
            matches = pattern.findall(response_text)
            if matches:
                # Show up to 3 examples of what matched
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
                    f'[HybridCritic] Invalid regex "{pattern}": {exc}. Skipping check.'
                )
        log_text = '\n\n'.join(log_lines) if log_lines else ''
        return failures, log_text

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _render_judge_prompt(
        self, best_candidate: str, oracle_response: str, history_text: str
    ) -> str:
        template = self._jinja_env.get_template('critic_judge.j2')
        return template.render(
            issue_text=self.issue_text,
            history_text=history_text,
            best_candidate_text=best_candidate,
            oracle_response_text=oracle_response,
            enable_symbolic_checks=self.enable_symbolic_checks,
        )

    def _render_recheck_prompt(
        self, failed_checks: list[dict], oracle_response: str, history_text: str
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
        parts = ['[CRITIC REVIEW — ORACLE RESPONSE REJECTED]\n']

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
        stage: str, prompt: str, response: str
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
            logger.warning(f'[HybridCritic] Failed to save prompt: {exc}')

    def _append_to_saved_prompt(
        self, step_index: int, attempt: int, stage: str, text: str
    ) -> None:
        """Append additional text (e.g. symbolic check results) to an already-saved prompt file."""
        save_dir = self._save_prompts_dir
        if not save_dir:
            return
        filename = f'step_{step_index:04d}_attempt_{attempt:02d}_{stage}.txt'
        path = os.path.join(save_dir, filename)
        try:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(text)
        except Exception as exc:
            logger.warning(f'[HybridCritic] Failed to append to prompt file: {exc}')

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        issue_text: str,
    ) -> 'HybridCritic | None':
        """Create HybridCritic from environment configuration."""
        from openhands.core.config.utils import get_llm_config_arg
        from openhands.llm.llm import LLM
        from openhands.llm.metrics import Metrics

        config_name = os.environ.get('GUIDED_CRITIC_LLM_CONFIG', 'blinded_critic')
        config_file = os.environ.get('CONFIG_FILE', 'config.toml')
        llm_config = get_llm_config_arg(config_name, config_file)
        if llm_config is None:
            logger.warning(
                f'[HybridCritic] No LLM config found for "{config_name}" in {config_file}. '
                'Critic disabled.'
            )
            return None

        llm_config.log_completions = False
        metrics = Metrics(model_name=llm_config.model)
        try:
            llm = LLM(config=llm_config, service_id='blinded_critic', metrics=metrics)
        except Exception as exc:
            logger.warning(f'[HybridCritic] Failed to init LLM: {exc}. Critic disabled.')
            return None

        logger.info(
            f'[HybridCritic] Initialized with model={llm_config.model} config={config_name}'
        )
        return cls(llm=llm, issue_text=issue_text)
