"""Sufficiency Critic for the Oracle Guided V2 agent.

Validates that a response is safe to make given the current state of
fact/artifact usage. Enforces one hard rule programmatically (no repro
creation when repro artifact unused) and delegates nuanced judgment
(code edits, premature conclusions) to an LLM.
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
                    f'[SufficiencyCritic] Transient LLM error '
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
class SufficiencyCriticResult:
    step_index: int
    passed: bool
    hard_rule_failed: bool = False   # True if the hard rule triggered (before LLM)
    reason: str = ''
    relevant_unused_facts: list[str] = field(default_factory=list)
    mark_used: list[str] = field(default_factory=list)  # artifact IDs to mark as used
    feedback: str = ''
    raw_response: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'step_index': self.step_index,
            'passed': self.passed,
            'hard_rule_failed': self.hard_rule_failed,
            'reason': self.reason,
            'relevant_unused_facts': self.relevant_unused_facts,
            'mark_used': self.mark_used,
            'feedback': self.feedback,
            'raw_response': self.raw_response,
        }


# ---------------------------------------------------------------------------
# Sufficiency Critic
# ---------------------------------------------------------------------------

class SufficiencyCritic:
    """Evaluates whether a response is safe given unused facts/artifacts."""

    def __init__(self, llm: Any, issue_text: str) -> None:
        self.llm = llm
        self.issue_text = issue_text

        self.max_json_parse_retries = int(
            os.environ.get('GUIDED_V2_SUFFICIENCY_CRITIC_JSON_PARSE_MAX_RETRIES', '3')
        )

        prompts_dir = Path(__file__).parent / 'prompts'
        self._jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(prompts_dir)),
            undefined=jinja2.StrictUndefined,
        )

        self._save_prompts_dir: str | None = os.environ.get(
            'GUIDED_V2_SUFFICIENCY_CRITIC_SAVE_PROMPTS_DIR'
        )

    # ------------------------------------------------------------------
    # Hard rule: reproduction gate
    # ------------------------------------------------------------------

    @staticmethod
    def _has_file_creation(response_text: str) -> bool:
        """Return True if the response contains a file-creation action."""
        if re.search(
            r'str_replace_editor.*"command"\s*:\s*"create"',
            response_text,
            re.IGNORECASE | re.DOTALL,
        ):
            return True
        return False

    @staticmethod
    def _check_hard_rules(
        response_text: str,
        unused_artifacts: list[dict],
        unused_facts: list[dict],
    ) -> SufficiencyCriticResult | None:
        """Check programmatic hard rules. Returns a failed result if violated, None if OK."""
        # Rule 1: When reproduce_script artifact is NOT used, block file creation
        repro_artifacts = [
            a for a in unused_artifacts if a['node_type'] == 'reproduce_script'
        ]
        if repro_artifacts and SufficiencyCritic._has_file_creation(response_text):
            # Collect the unused fact IDs that the repro artifact depends on
            unused_fact_ids = {f['id'] for f in unused_facts}
            blocking_facts: list[dict] = []
            for art in repro_artifacts:
                for dep_id in art.get('depends_on', []):
                    if dep_id in unused_fact_ids:
                        # Find the full fact entry
                        for f in unused_facts:
                            if f['id'] == dep_id:
                                blocking_facts.append(f)
                                break

            feedback_parts = [
                'Do not create files yet. The investigation has not '
                'progressed far enough to synthesize a reproduction '
                'script.',
            ]
            if blocking_facts:
                feedback_parts.append(
                    '\n\nThe following investigation areas must be explored '
                    'before a reproduction script can be created:'
                )
                for bf in blocking_facts:
                    feedback_parts.append(f'  - {bf["statement"]}')
                # Suggest a concrete alternative action
                feedback_parts.append(
                    '\n\n## SUGGESTED ALTERNATIVE\n'
                    'Instead of creating a file, guide the solver to '
                    'explore one of the investigation areas listed above. '
                    'For example, view or grep for the relevant code, '
                    'run an existing test, or examine a related file. '
                    'Pick the most natural next exploration step.'
                )
            else:
                feedback_parts.append(
                    '\n\n## SUGGESTED ALTERNATIVE\n'
                    'Continue exploring the codebase — view relevant '
                    'source files, grep for related methods, or examine '
                    'how similar features are implemented.'
                )

            return SufficiencyCriticResult(
                step_index=-1,  # will be set by caller
                passed=False,
                hard_rule_failed=True,
                reason='Reproduction artifact has not been revealed yet, '
                       'but the response creates a file.',
                feedback='\n'.join(feedback_parts),
            )

        return None  # all hard rules passed

    # ------------------------------------------------------------------
    # Artifact equivalence check
    # ------------------------------------------------------------------

    def _check_artifact_equivalence(
        self,
        step_index: int,
        response_text: str,
        unused_artifacts: list[dict],
        attempt: int = 0,
    ) -> SufficiencyCriticResult | None:
        """Check if the response creates an artifact equivalent to an unused one.

        When a hard rule would block (e.g., file creation while repro is
        unused), this method asks the LLM whether the response's file
        content is functionally equivalent to the expected artifact.

        Returns a *passed* ``SufficiencyCriticResult`` with ``mark_used``
        populated if the response matches, or ``None`` if no match.
        """
        # Find artifacts that have content we can compare against
        candidates: list[dict] = []
        for art in unused_artifacts:
            if art['node_type'] == 'reproduce_script' and art.get('code'):
                candidates.append(art)
            elif art['node_type'] in ('issue_analysis', 'fix_plan') and art.get('text'):
                candidates.append(art)

        if not candidates:
            return None

        # Build a compact equivalence prompt
        artifact_sections: list[str] = []
        for art in candidates:
            art_type = art['node_type']
            art_id = art['id']
            if art_type == 'reproduce_script':
                content = art.get('code', '')
                artifact_sections.append(
                    f'### Artifact [{art_id}] (type: {art_type})\n'
                    f'Description: {art.get("description", "")}\n'
                    f'Expected code:\n```python\n{content}\n```'
                )
            else:
                content = art.get('text', '')
                artifact_sections.append(
                    f'### Artifact [{art_id}] (type: {art_type})\n'
                    f'Content:\n{content}'
                )

        prompt = (
            'You are evaluating whether a proposed response creates an '
            'artifact that is functionally equivalent to one of the expected '
            'artifacts below.\n\n'
            '# Issue Description\n\n'
            f'{self.issue_text}\n\n'
            '# Expected Artifacts\n\n'
            + '\n\n'.join(artifact_sections) +
            '\n\n# Proposed Response\n\n'
            f'```\n{response_text[:3000]}\n```\n\n'
            '# Task\n\n'
            'Does the proposed response create a file or output that is '
            'functionally equivalent to any of the expected artifacts above? '
            '"Equivalent" means it tests/reproduces the same issue, or '
            'performs the same analysis/plan — it does NOT need to be '
            'identical code, just achieve the same goal.\n\n'
            'Return a JSON object:\n'
            '```json\n'
            '{"equivalent": true, "matched_artifact_ids": ["id1"], '
            '"reason": "brief explanation"}\n'
            '```\n'
            'If no match, return:\n'
            '```json\n'
            '{"equivalent": false, "matched_artifact_ids": [], '
            '"reason": "brief explanation"}\n'
            '```'
        )

        try:
            response = _llm_call_with_transient_retry(
                self.llm,
                messages=[{'role': 'user', 'content': prompt}],
                response_format={'type': 'json_object'},
            )
            raw = response.choices[0].message.content or ''
            self._save_prompt(
                step_index, attempt, 0, 'artifact_equivalence',
                prompt, raw,
            )

            data = self._extract_json(raw)
            if data and data.get('equivalent', False):
                matched_ids = data.get('matched_artifact_ids', [])
                if not isinstance(matched_ids, list):
                    matched_ids = [str(matched_ids)]
                # Validate matched IDs against actual artifacts
                valid_ids = {a['id'] for a in candidates}
                matched_ids = [mid for mid in matched_ids if mid in valid_ids]
                if matched_ids:
                    reason = data.get('reason', 'Response matches expected artifact')
                    logger.info(
                        f'[SufficiencyCritic] Artifact equivalence matched: '
                        f'{matched_ids} — {reason[:120]}'
                    )
                    return SufficiencyCriticResult(
                        step_index=step_index,
                        passed=True,
                        reason=f'Artifact equivalence: {reason}',
                        mark_used=matched_ids,
                        raw_response=raw,
                    )

            reason = data.get('reason', '') if data else ''
            logger.info(
                f'[SufficiencyCritic] Artifact equivalence not matched: '
                f'{reason[:120]}'
            )
        except Exception as exc:
            logger.warning(
                f'[SufficiencyCritic] Artifact equivalence check failed: {exc}'
            )
            self._save_prompt(
                step_index, attempt, 0, 'artifact_equivalence',
                prompt, str(exc),
            )

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        step_index: int,
        response_text: str,
        tool_calls_text: str,
        used_facts: list[dict],
        unused_facts: list[dict],
        unused_artifacts: list[dict],
        attempt: int = 0,
    ) -> SufficiencyCriticResult:
        """Validate that the response is safe given unused facts/artifacts.

        Steps:
        1. Check hard rules (programmatic — no LLM call)
        2. If hard rules pass, call LLM for nuanced judgment
        """
        # Combine response text + tool calls for hard rule checking
        full_text = (response_text or '') + ' ' + (tool_calls_text or '')

        # --- Hard rules ---
        hard_result = self._check_hard_rules(full_text, unused_artifacts, unused_facts)
        if hard_result is not None:
            hard_result.step_index = step_index

            # Before rejecting, check if the response creates an artifact
            # equivalent to the unused one.  If so, let it pass and mark
            # the artifact as used.
            equiv_result = self._check_artifact_equivalence(
                step_index=step_index,
                response_text=full_text,
                unused_artifacts=unused_artifacts,
                attempt=attempt,
            )
            if equiv_result is not None:
                logger.info(
                    f'[SufficiencyCritic] Hard rule overridden at step '
                    f'{step_index}: response matches artifact(s) '
                    f'{equiv_result.mark_used}'
                )
                return equiv_result

            logger.info(
                f'[SufficiencyCritic] Hard rule failed at step {step_index}: '
                f'{hard_result.reason}'
            )
            self._save_prompt(
                step_index, attempt, 0, 'sufficiency_hard_rule',
                f'HARD RULE FAILED: {hard_result.reason}',
                hard_result.feedback,
            )
            return hard_result

        # --- LLM judgment ---
        # Skip LLM call if there are no unused facts/artifacts to check against
        if not unused_facts and not unused_artifacts:
            return SufficiencyCriticResult(
                step_index=step_index,
                passed=True,
                reason='No unused facts or artifacts remain.',
            )

        prompt = self._render_prompt(
            response_text=response_text,
            tool_calls_text=tool_calls_text,
            used_facts=used_facts,
            unused_facts=unused_facts,
            unused_artifacts=unused_artifacts,
        )

        raw_response = ''
        for retry in range(self.max_json_parse_retries + 1):
            try:
                response = _llm_call_with_transient_retry(
                    self.llm,
                    messages=[{'role': 'user', 'content': prompt}],
                    response_format={'type': 'json_object'},
                )
                raw_response = response.choices[0].message.content or ''
                self._save_prompt(
                    step_index, attempt, retry, 'sufficiency_judge',
                    prompt, raw_response,
                )

                data = self._extract_json(raw_response)
                if data is not None:
                    passed = data.get('passed', True)
                    reason = data.get('reason', '')
                    relevant = data.get('relevant_unused_facts', [])
                    feedback = data.get('feedback', '')

                    if not isinstance(relevant, list):
                        relevant = []
                    if not isinstance(feedback, str):
                        feedback = str(feedback) if feedback else ''

                    logger.info(
                        f'[SufficiencyCritic] Step {step_index}: '
                        f'passed={passed}, reason={reason[:120]}'
                    )

                    return SufficiencyCriticResult(
                        step_index=step_index,
                        passed=passed,
                        reason=reason,
                        relevant_unused_facts=relevant,
                        feedback=feedback,
                        raw_response=raw_response,
                    )

                logger.warning(
                    f'[SufficiencyCritic] JSON parse retry {retry + 1}/{self.max_json_parse_retries + 1}'
                )
            except Exception as exc:
                logger.warning(f'[SufficiencyCritic] LLM call error: {exc}')
                raw_response = str(exc)
                self._save_prompt(
                    step_index, attempt, retry, 'sufficiency_judge',
                    prompt, raw_response,
                )

        # Fail-open on parse failure — assume response is safe
        logger.warning(
            '[SufficiencyCritic] All parse retries exhausted. Fail-open (passed=True).'
        )
        return SufficiencyCriticResult(
            step_index=step_index,
            passed=True,
            reason='Fail-open: JSON parse failed',
            raw_response=raw_response,
        )

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _render_prompt(
        self,
        response_text: str,
        tool_calls_text: str,
        used_facts: list[dict],
        unused_facts: list[dict],
        unused_artifacts: list[dict],
    ) -> str:
        template = self._jinja_env.get_template('sufficiency_judge.j2')
        return template.render(
            issue_text=self.issue_text,
            response_text=response_text,
            tool_calls_text=tool_calls_text,
            used_facts=used_facts,
            unused_facts=unused_facts,
            unused_artifacts=unused_artifacts,
        )

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
                f.write(f'=== {stage.upper()} RESPONSE ===\n')
                f.write('=' * 80 + '\n\n')
                f.write(response)
        except Exception as exc:
            logger.warning(f'[SufficiencyCritic] Failed to save prompt: {exc}')

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        issue_text: str,
    ) -> 'SufficiencyCritic | None':
        """Create SufficiencyCritic from environment configuration."""
        from openhands.core.config.utils import get_llm_config_arg
        from openhands.llm.llm import LLM
        from openhands.llm.metrics import Metrics

        config_name = os.environ.get('GUIDED_V2_SUFFICIENCY_CRITIC_LLM_CONFIG', 'sufficiency_critic')
        config_file = os.environ.get('CONFIG_FILE', 'config.toml')
        llm_config = get_llm_config_arg(config_name, config_file)
        if llm_config is None:
            logger.warning(
                f'[SufficiencyCritic] No LLM config found for "{config_name}" '
                f'in {config_file}. Sufficiency critic disabled.'
            )
            return None

        llm_config.log_completions = False
        metrics = Metrics(model_name=llm_config.model)
        try:
            llm = LLM(config=llm_config, service_id='sufficiency_critic', metrics=metrics)
        except Exception as exc:
            logger.warning(f'[SufficiencyCritic] Failed to init LLM: {exc}. Critic disabled.')
            return None

        logger.info(
            f'[SufficiencyCritic] Initialized with model={llm_config.model} config={config_name}'
        )
        return cls(llm=llm, issue_text=issue_text)
