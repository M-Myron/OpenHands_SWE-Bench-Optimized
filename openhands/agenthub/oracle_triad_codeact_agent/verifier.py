"""History-grounded neuro-symbolic verifier for oracle planner proposals.

Replaces the one-shot ``OracleProposalCritic`` with a bounded 4-stage pipeline:

    Stage 1 — Claim & precondition extraction (LLM or programmatic fallback)
    Stage 2 — History-grounded retrieval from StructuredHistoryMemory
    Stage 3 — Deterministic symbolic rule evaluation
    Stage 4 — Verdict synthesis (deterministic when possible, LLM-assisted for
              ambiguous leakage cases)

The verifier has NO live filesystem access, NO shell commands, and only a
bounded number of retrieval operations over the interaction history.
"""

from __future__ import annotations

import json
import os
import time
import re
from dataclasses import dataclass, field
from typing import Any

from jinja2 import Environment, FileSystemLoader

from openhands.agenthub.oracle_triad_codeact_agent.claim_extractor import (
    ClaimExtractor,
    ExtractionResult,
    ProgrammaticClaimExtractor,
)
from openhands.agenthub.oracle_triad_codeact_agent.history_memory import (
    HistoryUnit,
    StructuredHistoryMemory,
)
from openhands.agenthub.oracle_triad_codeact_agent.symbolic_rules import (
    RuleResult,
    SymbolicRuleEngine,
)
from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM
from openhands.llm.metrics import Metrics


# ---------------------------------------------------------------------------
# Verdict data structure
# ---------------------------------------------------------------------------


@dataclass
class VerificationVerdict:
    """Final output of the 4-stage verification pipeline."""

    step_index: int
    verdict: str  # 'valid' | 'invalid' | 'uncertain'
    claims: list[dict] = field(default_factory=list)
    explicit_preconditions: list[dict] = field(default_factory=list)
    inferred_preconditions: list[dict] = field(default_factory=list)
    retrieval_queries: list[str] = field(default_factory=list)
    retrieved_unit_ids: list[int] = field(default_factory=list)
    rule_results: list[dict] = field(default_factory=list)
    failed_obligations: list[str] = field(default_factory=list)
    suspected_leakage: list[str] = field(default_factory=list)
    feedback_message: str = ''
    suggestion: str = ''
    raw_extraction_response: str = ''
    raw_synthesis_response: str = ''
    reason: str = ''
    # Timing metadata (not serialized to verdict JSON, but accessible by agent)
    _timing: dict = field(default_factory=dict, repr=False)

    @property
    def valid(self) -> bool:
        """Backward-compat bridge: ``True`` if verdict is ``'valid'``."""
        return self.verdict == 'valid'

    def to_dict(self) -> dict[str, Any]:
        return {
            'step_index': self.step_index,
            'verdict': self.verdict,
            'valid': self.valid,
            'reason': self.reason,
            'claims': self.claims,
            'explicit_preconditions': self.explicit_preconditions,
            'inferred_preconditions': self.inferred_preconditions,
            'retrieval_queries': self.retrieval_queries,
            'retrieved_unit_ids': self.retrieved_unit_ids,
            'rule_results': self.rule_results,
            'failed_obligations': self.failed_obligations,
            'suspected_leakage': self.suspected_leakage,
            'feedback_message': self.feedback_message,
            'suggestion': self.suggestion,
        }


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL)
_JSON_BARE_RE = re.compile(r'\{.*\}', re.DOTALL)


def _extract_json(text: str) -> str | None:
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1)
    m = _JSON_BARE_RE.search(text)
    if m:
        return m.group(0)
    return None


# ---------------------------------------------------------------------------
# HistoryGroundedVerifier
# ---------------------------------------------------------------------------


class HistoryGroundedVerifier:
    """Neuro-symbolic verifier operating over structured interaction history.

    Usage::

        verifier = HistoryGroundedVerifier.from_env(issue_text)
        memory = StructuredHistoryMemory.from_events(state.history)
        verdict = verifier.verify(step_index, proposal, memory, fact_preconditions)
    """

    PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'prompts')

    def __init__(self, llm: LLM, issue_text: str) -> None:
        self.llm = llm
        self.issue_text = issue_text

        # Use LLM extractor when available; fallback to programmatic
        use_programmatic_only = (
            os.environ.get('VERIFIER_PROGRAMMATIC_ONLY', '0') == '1'
        )
        if use_programmatic_only:
            self._claim_extractor: ClaimExtractor | ProgrammaticClaimExtractor = (
                ProgrammaticClaimExtractor()
            )
        else:
            self._claim_extractor = ClaimExtractor(llm)

        self._jinja_env = Environment(
            loader=FileSystemLoader(self.PROMPTS_DIR),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def verify(
        self,
        step_index: int,
        proposal_text: str,
        history_memory: StructuredHistoryMemory,
        fact_preconditions: list[dict] | None = None,
        attempt: int = 0,
    ) -> VerificationVerdict:
        """Run the full 4-stage verification pipeline."""

        verify_t0 = time.monotonic()
        vtiming: dict[str, float] = {}
        vllm_calls = 0

        # ---- Stage 1: Claim & precondition extraction --------------------
        t0 = time.monotonic()
        history_summary = history_memory.summary()

        if isinstance(self._claim_extractor, ClaimExtractor):
            extraction = self._claim_extractor.extract(
                proposal_text=proposal_text,
                oracle_preconditions=fact_preconditions,
                issue_text=self.issue_text,
                step_index=step_index,
                history_summary=history_summary,
            )
        else:
            extraction = self._claim_extractor.extract(
                proposal_text=proposal_text,
                oracle_preconditions=fact_preconditions,
                issue_text=self.issue_text,
            )

        raw_extraction = extraction.raw_llm_response
        raw_extraction_prompt = extraction.raw_llm_prompt

        self._maybe_save_prompt(
            step_index, attempt, 'extraction', raw_extraction_prompt, raw_extraction,
        )
        vtiming['stage1_extraction'] = time.monotonic() - t0
        vllm_calls += 1  # extraction is 1 LLM call (unless programmatic)

        # ---- Stage 2: History-grounded retrieval -------------------------
        t0 = time.monotonic()
        retrieved_units, retrieval_queries = self._run_retrieval(
            extraction, history_memory,
        )
        vtiming['stage2_retrieval'] = time.monotonic() - t0

        # ---- Stage 3: Symbolic rule evaluation ---------------------------
        t0 = time.monotonic()
        engine = SymbolicRuleEngine(history_memory, self.issue_text)
        rule_results = engine.evaluate_all(
            claims=extraction.claims,
            preconditions=extraction.all_preconditions,
            retrieved_units=retrieved_units,
            fact_preconditions=fact_preconditions,
        )
        vtiming['stage3_symbolic'] = time.monotonic() - t0

        # Check for deterministic (non-LLM-assist) high-severity failures first.
        # If present, reject immediately without spending LLM calls.
        deterministic_high = engine.get_failed_high_excluding_llm_assist(rule_results)
        if deterministic_high:
            # Skip LLM resolution — deterministic reject
            failed_high = deterministic_high
            uncertain = engine.get_uncertain(rule_results)
            vtiming['stage3_5_llm_resolution'] = 0.0
        else:
            # ---- Stage 3.5: LLM-assisted rule resolution -----------------
            t0 = time.monotonic()
            llm_rules_count = len(SymbolicRuleEngine.get_llm_assist_needed(rule_results))
            rule_results = self._resolve_llm_assisted_rules(
                rule_results=rule_results,
                proposal_text=proposal_text,
                retrieved_units=retrieved_units,
                step_index=step_index,
                attempt=attempt,
            )
            vtiming['stage3_5_llm_resolution'] = time.monotonic() - t0
            if llm_rules_count > 0:
                vllm_calls += 1  # single batched call for all rules
            # Recalculate after LLM resolution (all llm_assist flags cleared)
            failed_high = engine.get_failed_high(rule_results)
            uncertain = engine.get_uncertain(rule_results)

        # ---- Stage 4: Verdict synthesis ----------------------------------
        t0 = time.monotonic()
        verdict, reason, suspected_leakage, suggestion, raw_synthesis = (
            self._synthesize_verdict(
                step_index=step_index,
                proposal_text=proposal_text,
                extraction=extraction,
                rule_results=rule_results,
                failed_high=failed_high,
                uncertain=uncertain,
                retrieved_units=retrieved_units,
                attempt=attempt,
            )
        )
        vtiming['stage4_synthesis'] = time.monotonic() - t0
        if raw_synthesis:
            vllm_calls += 1  # synthesis made an LLM call

        vtiming['verify_total'] = time.monotonic() - verify_t0

        logger.info(
            f'[HistoryGroundedVerifier] Step {step_index} attempt {attempt}: '
            f'extraction={vtiming["stage1_extraction"]:.1f}s, '
            f'retrieval={vtiming["stage2_retrieval"]:.2f}s, '
            f'symbolic={vtiming["stage3_symbolic"]:.2f}s, '
            f'llm_resolution={vtiming["stage3_5_llm_resolution"]:.1f}s, '
            f'synthesis={vtiming["stage4_synthesis"]:.1f}s, '
            f'total={vtiming["verify_total"]:.1f}s ({vllm_calls} LLM calls) '
            f'verdict={verdict}'
        )

        # Build feedback message for planner retry
        feedback = self._build_feedback(
            verdict, reason, failed_high, uncertain,
            rule_results, suspected_leakage, suggestion, extraction,
        )

        # Build failed obligations list
        failed_obligations: list[str] = []
        for r in rule_results:
            if not r.passed and r.severity in ('high', 'medium'):
                failed_obligations.append(f'{r.rule_id}: {r.reason[:100]}')

        final_verdict = VerificationVerdict(
            step_index=step_index,
            verdict=verdict,
            reason=reason,
            claims=[c.to_dict() for c in extraction.claims],
            explicit_preconditions=[p.to_dict() for p in extraction.explicit_preconditions],
            inferred_preconditions=[p.to_dict() for p in extraction.inferred_preconditions],
            retrieval_queries=retrieval_queries,
            retrieved_unit_ids=[u.unit_id for u in retrieved_units],
            rule_results=[r.to_dict() for r in rule_results],
            failed_obligations=failed_obligations,
            suspected_leakage=suspected_leakage,
            feedback_message=feedback,
            suggestion=suggestion,
            raw_extraction_response=raw_extraction,
            raw_synthesis_response=raw_synthesis,
            _timing={**vtiming, 'llm_calls': vllm_calls},
        )

        # Save the full verdict as a prompt log for debugging
        self._maybe_save_verdict(
            step_index, attempt, final_verdict,
        )

        return final_verdict

    # ------------------------------------------------------------------
    # Stage 2: Retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _run_retrieval(
        extraction: ExtractionResult,
        memory: StructuredHistoryMemory,
    ) -> tuple[list[HistoryUnit], list[str]]:
        """Execute the retrieval plan from the extraction result."""
        all_units: dict[int, HistoryUnit] = {}
        queries_used: list[str] = []

        for query in extraction.retrieval_plan:
            queries_used.append(query)

            if query.startswith('file:'):
                path = query[5:]
                for u in memory.file_path_search(path):
                    all_units[u.unit_id] = u

            elif query.startswith('phase:'):
                phase = query[6:]
                for u in memory.phase_search(phase):
                    all_units[u.unit_id] = u

            elif query.startswith('keyword:'):
                keywords = [k.strip() for k in query[8:].split(',') if k.strip()]
                for u in memory.keyword_search(keywords, top_k=5):
                    all_units[u.unit_id] = u

            elif query.startswith('tag:'):
                tag = query[4:]
                for u in memory.tag_search({tag}):
                    all_units[u.unit_id] = u

            elif query.startswith('symbol:'):
                sym = query[7:]
                for u in memory.keyword_search([sym], top_k=3):
                    all_units[u.unit_id] = u

        # Also retrieve units for files/symbols from claims (derived retrieval)
        for claim in extraction.claims:
            for fp in claim.file_paths[:3]:
                for u in memory.file_path_search(fp):
                    all_units[u.unit_id] = u
            for sym in claim.symbols[:3]:
                for u in memory.keyword_search([sym], top_k=2):
                    all_units[u.unit_id] = u

        # Sort by unit_id for deterministic ordering
        sorted_units = sorted(all_units.values(), key=lambda u: u.unit_id)
        return sorted_units, queries_used

    # ------------------------------------------------------------------
    # Stage 3.5: LLM-assisted rule resolution
    # ------------------------------------------------------------------

    def _resolve_llm_assisted_rules(
        self,
        rule_results: list[RuleResult],
        proposal_text: str,
        retrieved_units: list[HistoryUnit],
        step_index: int,
        attempt: int,
    ) -> list[RuleResult]:
        """Resolve rules that failed symbolically but are marked for LLM adjudication.

        For rules with ``needs_llm_assist=True``, we make a **single batched**
        LLM call using the ``resolve_rules_batch.j2`` template to resolve all
        failing rules at once. If the LLM overrules a symbolic failure, we flip
        the result to passed.

        Returns a NEW list of RuleResult (original list is not mutated).
        """
        llm_rules = SymbolicRuleEngine.get_llm_assist_needed(rule_results)
        if not llm_rules:
            return rule_results

        # Build evidence snippets once (shared across all resolutions)
        evidence_snippets = [
            {
                'unit_id': u.unit_id,
                'action_type': u.action_type or 'unknown',
                'phase_hint': u.phase_hint or 'unknown',
                'action_summary': u.action_summary or '',
                'text_snippet': (
                    (u.observation_text or u.action_text or '')
                ),
            }
            for u in retrieved_units
        ]

        # Render a single batched prompt for ALL failing rules
        template = self._jinja_env.get_template('resolve_rules_batch.j2')
        prompt = template.render(
            rules=llm_rules,
            proposal_text=proposal_text,
            issue_text=self.issue_text,
            evidence_snippets=evidence_snippets,
        )

        max_resolve_retries = 2
        data = None

        for resolve_try in range(max_resolve_retries + 1):
            try:
                response = self.llm.completion(
                    messages=[{'role': 'user', 'content': prompt}],
                )
                raw_text = response.choices[0].message.content or ''
            except Exception as exc:
                if resolve_try < max_resolve_retries:
                    logger.warning(
                        f'[HistoryGroundedVerifier] Batch resolution LLM call failed '
                        f'(attempt {resolve_try + 1}): {exc}. Retrying...'
                    )
                    continue
                logger.warning(
                    f'[HistoryGroundedVerifier] Batch resolution failed after '
                    f'{max_resolve_retries + 1} attempts: {exc}. Fail-open override for all rules.'
                )
                raw_text = ''
                break

            self._maybe_save_prompt(
                step_index, attempt, 'resolve_batch', prompt, raw_text,
            )

            json_str = _extract_json(raw_text)
            if json_str is not None:
                try:
                    data = json.loads(json_str)
                    if 'results' in data and isinstance(data['results'], list):
                        break
                    data = None
                except json.JSONDecodeError:
                    data = None

            if resolve_try < max_resolve_retries:
                logger.warning(
                    f'[HistoryGroundedVerifier] Unparseable batch response '
                    f'(attempt {resolve_try + 1}). Retrying...'
                )
            else:
                logger.warning(
                    f'[HistoryGroundedVerifier] Batch resolution unparseable after '
                    f'{max_resolve_retries + 1} attempts. Fail-open override for all rules.'
                )

        # Parse results into a map: rule_id -> {verdict, reason}
        llm_verdicts: dict[str, dict] = {}
        if data and 'results' in data:
            for entry in data['results']:
                rid = str(entry.get('rule_id', ''))
                if rid:
                    llm_verdicts[rid] = {
                        'verdict': str(entry.get('verdict', '')).lower(),
                        'reason': str(entry.get('reason', '')),
                    }

        # Build resolved RuleResult map
        resolved: dict[str, RuleResult] = {}
        for rule in llm_rules:
            v = llm_verdicts.get(rule.rule_id)
            if v is None:
                # Rule not found in LLM response → fail-open
                resolved[rule.rule_id] = RuleResult(
                    rule_id=rule.rule_id,
                    rule_family=rule.rule_family,
                    passed=True,
                    severity=rule.severity,
                    reason='Rule not in batch LLM response; fail-open override.',
                    needs_llm_assist=False,
                    llm_context={},
                )
            elif v['verdict'] == 'overruled':
                resolved[rule.rule_id] = RuleResult(
                    rule_id=rule.rule_id,
                    rule_family=rule.rule_family,
                    passed=True,
                    severity=rule.severity,
                    reason=f'LLM overruled: {v["reason"]}',
                    needs_llm_assist=False,
                    llm_context={},
                )
            else:
                resolved[rule.rule_id] = RuleResult(
                    rule_id=rule.rule_id,
                    rule_family=rule.rule_family,
                    passed=False,
                    severity=rule.severity,
                    reason=f'{rule.reason} [LLM confirmed: {v["reason"]}]',
                    needs_llm_assist=False,
                    llm_context={},
                )

        # Rebuild the full result list, replacing resolved rules
        new_results: list[RuleResult] = []
        for r in rule_results:
            if r.rule_id in resolved:
                new_results.append(resolved[r.rule_id])
            else:
                new_results.append(r)

        return new_results

    # ------------------------------------------------------------------
    # Stage 4: Verdict synthesis
    # ------------------------------------------------------------------

    def _synthesize_verdict(
        self,
        step_index: int,
        proposal_text: str,
        extraction: ExtractionResult,
        rule_results: list[RuleResult],
        failed_high: list[RuleResult],
        uncertain: list[RuleResult],
        retrieved_units: list[HistoryUnit],
        attempt: int,
    ) -> tuple[str, str, list[str], str, str]:
        """Determine verdict. Returns (verdict, reason, leakage, suggestion, raw_response)."""

        # Fast path: deterministic verdicts without LLM call
        if failed_high:
            reasons = [f'{r.rule_id}: {r.reason}' for r in failed_high]
            reason = f'Rejected by {len(failed_high)} high-severity rule(s): {"; ".join(reasons[:3])}'
            suggestion = self._build_deterministic_suggestion(failed_high, extraction)
            return 'invalid', reason, [], suggestion, ''

        # If no failures at all → valid
        all_passed = all(r.passed for r in rule_results)
        if all_passed and not uncertain:
            return 'valid', 'All symbolic rules passed.', [], '', ''

        # Ambiguous case: uncertain rules (typically C-family leakage) → LLM synthesis
        if uncertain:
            return self._llm_synthesis(
                step_index, proposal_text, extraction,
                rule_results, failed_high, uncertain,
                retrieved_units, attempt,
            )

        # Medium/low failures only, no uncertain → valid with warnings
        warnings = [r for r in rule_results if not r.passed]
        warning_reasons = [f'{r.rule_id} ({r.severity})' for r in warnings]
        return (
            'valid',
            f'Passed with {len(warnings)} warning(s): {", ".join(warning_reasons[:5])}',
            [],
            '',
            '',
        )

    def _llm_synthesis(
        self,
        step_index: int,
        proposal_text: str,
        extraction: ExtractionResult,
        rule_results: list[RuleResult],
        failed_high: list[RuleResult],
        uncertain: list[RuleResult],
        retrieved_units: list[HistoryUnit],
        attempt: int,
    ) -> tuple[str, str, list[str], str, str]:
        """Call LLM for verdict synthesis on ambiguous cases."""
        evidence_summaries = [
            {
                'unit_id': u.unit_id,
                'action_type': u.action_type or 'unknown',
                'phase_hint': u.phase_hint or 'unknown',
                'action_summary': u.action_summary,
            }
            for u in retrieved_units[:20]
        ]

        template = self._jinja_env.get_template('synthesize_verdict.j2')
        prompt = template.render(
            step_index=step_index,
            proposal_text=proposal_text,
            issue_text=self.issue_text,
            failed_high=[r.to_dict() for r in failed_high],
            uncertain_rules=[r.to_dict() for r in uncertain],
            all_results=[r.to_dict() for r in rule_results],
            claims=[c.to_dict() for c in extraction.claims],
            evidence_summaries=evidence_summaries,
        )

        try:
            response = self.llm.completion(
                messages=[{'role': 'user', 'content': prompt}],
            )
            raw_text = response.choices[0].message.content or ''
        except Exception as exc:
            logger.warning(
                f'[HistoryGroundedVerifier] Synthesis LLM call failed: {exc}. '
                'Treating uncertain cases as valid (fail-open).'
            )
            return (
                'uncertain',
                f'Synthesis LLM failed ({exc}); treating as valid.',
                [],
                '',
                str(exc),
            )

        self._maybe_save_prompt(step_index, attempt, 'synthesis', prompt, raw_text)

        # Parse synthesis response
        json_str = _extract_json(raw_text)
        if json_str is None:
            return (
                'uncertain',
                'Synthesis response not parseable; treating as valid.',
                [],
                '',
                raw_text,
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return (
                'uncertain',
                'Synthesis JSON malformed; treating as valid.',
                [],
                '',
                raw_text,
            )

        verdict = str(data.get('verdict', 'uncertain')).lower()
        if verdict not in ('valid', 'invalid', 'uncertain'):
            verdict = 'uncertain'

        reason = str(data.get('reason', ''))
        suspected = [str(x) for x in data.get('suspected_leakage', [])]
        suggestion = str(data.get('suggestion', ''))

        return verdict, reason, suspected, suggestion, raw_text

    @staticmethod
    def _build_deterministic_suggestion(
        failed_high: list[RuleResult],
        extraction: ExtractionResult,
    ) -> str:
        """Build a concrete suggestion from deterministic failures."""
        suggestions: list[str] = []
        for r in failed_high:
            if 'analysis' in r.rule_id.lower() or 'A1' in r.rule_id:
                suggestions.append(
                    'Use the think tool to perform explicit fix analysis (Phase 5) '
                    'stating what the bug is, where it is, why the code is wrong, '
                    'and how to fix it — BEFORE proposing any code edit.'
                )
            elif 'file_path' in r.rule_id.lower() or 'B1' in r.rule_id:
                suggestions.append(
                    'Use grep or find to discover the file path in the codebase '
                    'before referencing it.'
                )
            elif 'edit_target' in r.rule_id.lower() or 'B3' in r.rule_id:
                suggestions.append(
                    'Read the target file before proposing an edit.'
                )
            elif 'verification' in r.rule_id.lower() or 'A2' in r.rule_id:
                suggestions.append(
                    'Implement the code fix (Phase 6) before running verification tests.'
                )
            elif 'finalization' in r.rule_id.lower() or 'A3' in r.rule_id:
                suggestions.append(
                    'Run verification tests (Phase 7) before final review.'
                )
            elif 'oracle' in r.rule_id.lower() or 'C3' in r.rule_id:
                suggestions.append(
                    'Ensure all referenced files and symbols have public provenance '
                    'in the interaction history.'
                )

        # Deduplicate
        seen: set[str] = set()
        unique: list[str] = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        return ' '.join(unique) if unique else 'Revise the proposal to address the failed rules.'

    # ------------------------------------------------------------------
    # Feedback message construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_feedback(
        verdict: str,
        reason: str,
        failed_high: list[RuleResult],
        uncertain: list[RuleResult],
        all_results: list[RuleResult],
        suspected_leakage: list[str],
        suggestion: str,
        extraction: ExtractionResult,
    ) -> str:
        """Build structured feedback for planner retry."""
        if verdict == 'valid':
            return ''

        lines: list[str] = [
            '[VERIFIER REVIEW - PROPOSAL REJECTED]',
            '',
            f'Verdict: {verdict}',
            f'Reason: {reason}',
        ]

        if failed_high:
            lines.append('')
            lines.append('Failed verification rules (high severity):')
            for r in failed_high:
                lines.append(f'  - [{r.rule_id}] {r.reason[:150]}')

        failed_medium = [r for r in all_results if not r.passed and r.severity == 'medium']
        if failed_medium:
            lines.append('')
            lines.append('Additional warnings (medium severity):')
            for r in failed_medium:
                lines.append(f'  - [{r.rule_id}] {r.reason[:120]}')

        if suspected_leakage:
            lines.append('')
            lines.append('Suspected leakage/unjustified knowledge:')
            for item in suspected_leakage:
                lines.append(f'  - {item}')

        if suggestion:
            lines.append('')
            lines.append(f'Suggested next action: {suggestion}')

        lines.extend([
            '',
            'Revise the proposal with these constraints:',
            '  1. Do not mention or imply oracle/golden patch/test details.',
            '  2. Keep all claims grounded in the observed interaction history.',
            '  3. Follow workflow phase ordering strictly.',
            '  4. If a file/symbol has not been observed, discover it first.',
        ])

        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Prompt saving
    # ------------------------------------------------------------------

    @staticmethod
    def _maybe_save_prompt(
        step_index: int,
        attempt: int,
        stage: str,
        prompt: str,
        raw_response: str,
    ) -> None:
        save_dir = os.environ.get('ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS_DIR', '')
        if not save_dir:
            return
        try:
            os.makedirs(save_dir, exist_ok=True)
            fname = f'step_{step_index:04d}_attempt_{attempt:02d}_{stage}.txt'
            fpath = os.path.join(save_dir, fname)
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(f'=== VERIFIER {stage.upper()} PROMPT ===\n')
                f.write(prompt if prompt else '(prompt not captured)')
                f.write(f'\n\n=== VERIFIER {stage.upper()} RESPONSE ===\n')
                f.write(raw_response)
                f.write('\n')
        except Exception as exc:
            logger.warning(f'[HistoryGroundedVerifier] Failed to save prompt: {exc}')

    @staticmethod
    def _maybe_save_verdict(
        step_index: int,
        attempt: int,
        verdict: 'VerificationVerdict',
    ) -> None:
        """Save the full verdict as a structured JSON file for debugging."""
        save_dir = os.environ.get('ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS_DIR', '')
        if not save_dir:
            return
        try:
            os.makedirs(save_dir, exist_ok=True)
            fname = f'step_{step_index:04d}_attempt_{attempt:02d}_verdict.json'
            fpath = os.path.join(save_dir, fname)
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(verdict.to_dict(), f, indent=2, ensure_ascii=False)
                f.write('\n')
        except Exception as exc:
            logger.warning(f'[HistoryGroundedVerifier] Failed to save verdict: {exc}')

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, issue_text: str) -> 'HistoryGroundedVerifier | None':
        """Create a verifier from environment configuration.

        Uses the same LLM config key as the legacy critic for backward compat.
        """
        from openhands.core.config.utils import get_llm_config_arg

        config_name = os.environ.get(
            'VERIFIER_LLM_CONFIG',
            os.environ.get('ORACLE_PROPOSAL_CRITIC_LLM_CONFIG', 'blinded_critic'),
        )
        config_file = os.environ.get('CONFIG_FILE', 'config.toml')
        llm_config = get_llm_config_arg(config_name, config_file)
        if llm_config is None:
            logger.warning(
                f'[HistoryGroundedVerifier] No LLM config for "{config_name}" '
                f'in {config_file}. Verifier disabled.'
            )
            return None

        llm_config.log_completions = False
        metrics = Metrics(model_name=llm_config.model)
        try:
            llm = LLM(
                config=llm_config,
                service_id='history_grounded_verifier',
                metrics=metrics,
            )
        except Exception as exc:
            logger.warning(
                f'[HistoryGroundedVerifier] LLM init failed: {exc}. Verifier disabled.'
            )
            return None

        logger.info(
            f'[HistoryGroundedVerifier] Initialized with model={llm_config.model} '
            f'config={config_name}'
        )
        return cls(llm=llm, issue_text=issue_text)
