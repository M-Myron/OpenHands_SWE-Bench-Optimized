"""Blinded Critic — validates agent step responses for reachability and non-leakage.

The critic is "blinded" because it never sees the golden patch or golden test.
It only receives:
  - The original issue description
  - The accumulated action/observation history up to this step
  - The agent's current proposed response (reasoning + intended action)

It judges whether the proposed response could have been derived purely from
the available evidence (prior observations), or whether it relies on information
that could not yet be known — which would indicate leakage from the golden answer.
"""

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
class ValidationResult:
    """Result from the Blinded Critic for a single agent step."""

    step_index: int
    valid: bool
    reason: str
    # Knowledge the agent used that wasn't yet available from observations
    unjustified_knowledge: list[str] = field(default_factory=list)
    # Conditions that must hold / steps that must happen before this action is justified
    prerequisite_conditions: list[str] = field(default_factory=list)
    # Feedback to send back to the main agent on failure
    feedback_message: str = ''
    # The agent's full response that was being judged (thought + tool calls)
    agent_response_text: str = ''
    # Raw critic response for debugging
    raw_critic_response: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'step_index': self.step_index,
            'valid': self.valid,
            'reason': self.reason,
            'unjustified_knowledge': self.unjustified_knowledge,
            'prerequisite_conditions': self.prerequisite_conditions,
            'feedback_message': self.feedback_message,
            'agent_response_text': self.agent_response_text,
            'raw_critic_response': self.raw_critic_response,
        }


class BlindedCritic:
    """A second LLM that validates whether each agent step is reachable
    from accumulated evidence, enforcing non-leakage of the golden answer.

    The critic is initialized with only the issue text (no golden patch/test).
    On each call to :meth:`validate`, it receives the sanitized history and the
    agent's current response and returns a :class:`ValidationResult`.
    """

    PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'prompts')

    def __init__(self, llm: LLM, issue_text: str) -> None:
        """
        Args:
            llm: An LLM instance configured for the Blinded Critic
                 (e.g. from ``[llm.blinded_critic]`` in config.toml).
            issue_text: The plain issue description visible to the main agent,
                        WITHOUT any golden patch / golden test content.
        """
        self.llm = llm
        self.issue_text = issue_text
        self._jinja_env = Environment(
            loader=FileSystemLoader(self.PROMPTS_DIR),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        step_index: int,
        history_text: str,
        agent_response_text: str,
        attempt: int = 0,
    ) -> ValidationResult:
        """Validate a single agent step.

        Args:
            step_index: 0-based index of the current step in the trajectory.
            history_text: Human-readable summary of all prior actions and
                          observations (golden data already stripped).
            agent_response_text: The agent's full response for this step —
                                 reasoning/thought + intended action — as a
                                 plain-text string.
            attempt: Retry attempt number within this step (0-based).  Used
                     when saving debug prompts to disk.

        Returns:
            A :class:`ValidationResult` with the critic's verdict.
        """
        prompt = self._render_prompt(step_index, history_text, agent_response_text)

        try:
            response = self.llm.completion(
                messages=[{'role': 'user', 'content': prompt}],
            )
            raw_text: str = response.choices[0].message.content or ''
        except Exception as e:
            logger.warning(f'[BlindedCritic] LLM call failed: {e}. Allowing step.')
            self._maybe_save_prompt(step_index, attempt, prompt, f'[LLM CALL FAILED]: {e}')
            return ValidationResult(
                step_index=step_index,
                valid=True,
                reason=f'Critic LLM call failed ({e}); step allowed by default.',
                agent_response_text=agent_response_text,
                raw_critic_response='',
            )

        self._maybe_save_prompt(step_index, attempt, prompt, raw_text)
        result = BlindedCritic._parse_response(step_index, raw_text)
        result.agent_response_text = agent_response_text
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _maybe_save_prompt(
        step_index: int,
        attempt: int,
        prompt: str,
        raw_response: str,
    ) -> None:
        """If ``BLINDED_CRITIC_SAVE_PROMPTS_DIR`` is set, write the full
        critic prompt and raw response to a text file for offline debugging.

        Enable by setting the env var at run time (see ``run_guided_infer.sh``):

        .. code-block:: bash

            export BLINDED_CRITIC_SAVE_PROMPTS=1

        The per-instance save directory is set automatically by
        :func:`run_infer_guided.process_instance_guided` to::

            {eval_output_dir}/blinded_critic_prompts/{instance_id}/

        Files are named ``step_{step_index:04d}_attempt_{attempt:02d}.txt``.
        """
        save_dir = os.environ.get('BLINDED_CRITIC_SAVE_PROMPTS_DIR', '')
        if not save_dir:
            return
        try:
            os.makedirs(save_dir, exist_ok=True)
            fname = f'step_{step_index:04d}_attempt_{attempt:02d}.txt'
            fpath = os.path.join(save_dir, fname)
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write('=== CRITIC PROMPT ===\n')
                f.write(prompt)
                f.write('\n\n=== CRITIC RAW RESPONSE ===\n')
                f.write(raw_response)
                f.write('\n')
        except Exception as exc:
            logger.warning(f'[BlindedCritic] Failed to save prompt debug file: {exc}')

    def _render_prompt(
        self, step_index: int, history_text: str, agent_response_text: str
    ) -> str:
        template = self._jinja_env.get_template('validate_response.j2')
        return template.render(
            issue_text=self.issue_text,
            step_index=step_index,
            history_text=history_text,
            agent_response_text=agent_response_text,
        )

    @staticmethod
    def _parse_response(step_index: int, raw_text: str) -> ValidationResult:
        """Parse the critic's JSON response into a :class:`ValidationResult`."""
        # Try to extract a JSON block
        json_str = BlindedCritic._extract_json(raw_text)
        if json_str is None:
            # Fallback: interpret as text; if it says "valid" allow by default
            logger.warning(
                '[BlindedCritic] Could not find JSON in critic response; '
                'defaulting to valid=True.'
            )
            return ValidationResult(
                step_index=step_index,
                valid=True,
                reason='Critic response not parseable as JSON; step allowed.',
                raw_critic_response=raw_text,
            )

        try:
            data: dict[str, Any] = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f'[BlindedCritic] JSON parse error: {e}. Allowing step.')
            return ValidationResult(
                step_index=step_index,
                valid=True,
                reason=f'Critic JSON malformed ({e}); step allowed.',
                raw_critic_response=raw_text,
            )

        unjustified: list[str] = [
            str(x) for x in data.get('unjustified_knowledge', [])
        ]
        prereqs: list[str] = [
            str(x) for x in data.get('prerequisite_conditions', [])
        ]
        reason: str = str(data.get('reason', ''))
        # Enforce invariant: non-empty unjustified_knowledge always means invalid,
        # regardless of whatever 'valid' value the LLM returned.
        if unjustified:
            valid = False
        else:
            valid = bool(data.get('valid', True))

        feedback_message = BlindedCritic._build_feedback_message(
            valid, reason, unjustified, prereqs
        )

        return ValidationResult(
            step_index=step_index,
            valid=valid,
            reason=reason,
            unjustified_knowledge=unjustified,
            prerequisite_conditions=prereqs,
            feedback_message=feedback_message,
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
        parts = [
            '[QA REVIEW — ACTION REJECTED]',
            '',
            f'Reason: {reason}',
        ]
        if unjustified:
            parts += ['', 'Information used that is NOT yet derivable from observations:']
            parts += [f'  - {item}' for item in unjustified]
        if prereqs:
            parts += ['', 'Prerequisite steps / conditions that must be established first:']
            parts += [f'  - {item}' for item in prereqs]
        parts += [
            '',
            'IMPORTANT — How to fix your response:',
            '  1. Do NOT mention the reference patch, reference test, or any information',
            '     from them. Pretend they do not exist in your response text.',
            '  2. You MAY reference names that appear in the issue description (class names,',
            '     function names, error messages — the problem statement is yours to use).',
            '     What you must NOT do is assume you know their exact file path, import',
            '     path, line number, or implementation details without first exploring.',
            '     For example: saying "I will look for separability_matrix" is fine;',
            '     writing "from astropy.modeling.separable import separability_matrix"',
            '     without having read any source file is NOT fine.',
            '  3. Your NEXT action must be a concrete exploration tool call (e.g. read a',
            '     file, run grep, execute a script) that produces the observation you need.',
            '  4. After the tool call returns, you may reason from THAT output — not from',
            '     prior knowledge.',
            '  5. Keep your thought text minimal: state only what you are about to do and',
            '     why it follows from observations already made. Do NOT pre-announce',
            '     specific import paths, exact values, or fixes before you have read the code.',
            '',
            'In short: explore first, then conclude. Every claim about implementation',
            'details must cite a tool output from this session as its evidence.',
        ]
        return '\n'.join(parts)

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Extract the first JSON object from text, handling markdown code blocks."""
        # Try ```json ... ``` fence
        fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if fence:
            return fence.group(1)
        # Try bare {...}
        bare = re.search(r'\{.*\}', text, re.DOTALL)
        if bare:
            return bare.group(0)
        return None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, issue_text: str) -> 'BlindedCritic | None':
        """Construct a :class:`BlindedCritic` from environment / config.

        Reads the llm config named by the env var
        ``BLINDED_CRITIC_LLM_CONFIG`` (default: ``blinded_critic``) from
        ``config.toml``.  Returns ``None`` if no config can be found so the
        agent can fall back to running without validation.
        """
        from openhands.core.config.utils import get_llm_config_arg

        config_name = os.environ.get('BLINDED_CRITIC_LLM_CONFIG', 'blinded_critic')
        config_file = os.environ.get('CONFIG_FILE', 'config.toml')
        llm_config = get_llm_config_arg(config_name, config_file)

        if llm_config is None:
            logger.warning(
                f'[BlindedCritic] No LLM config found for "{config_name}" in {config_file}. '
                'Step validation will be disabled.'
            )
            return None

        # Disable completion logging for the critic to avoid clutter
        llm_config.log_completions = False

        metrics = Metrics(model_name=llm_config.model)
        try:
            llm = LLM(config=llm_config, service_id='blinded_critic', metrics=metrics)
        except Exception as e:
            logger.warning(
                f'[BlindedCritic] Failed to initialise LLM from config "{config_name}": {e}. '
                'Step validation will be disabled.'
            )
            return None

        logger.info(
            f'[BlindedCritic] Initialised with model={llm_config.model} '
            f'config_name={config_name}'
        )
        return cls(llm=llm, issue_text=issue_text)
