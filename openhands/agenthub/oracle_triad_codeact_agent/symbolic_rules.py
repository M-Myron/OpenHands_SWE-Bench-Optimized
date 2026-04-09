"""Symbolic verification rules for the neuro-symbolic verifier.

All rules in this module are deterministic — no LLM calls.  Each rule
evaluates claims and preconditions against retrieved evidence from
``StructuredHistoryMemory`` and returns a ``RuleResult``.

Rule families:
    A — Workflow phase ordering
    B — Reachability (file paths, symbols, edit targets)
    C — Leakage detection
    D — Evidence sufficiency
    E — Discoverability advisories
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from openhands.agenthub.oracle_triad_codeact_agent.claim_extractor import (
    Claim,
    Precondition,
)
from openhands.agenthub.oracle_triad_codeact_agent.history_memory import (
    HistoryUnit,
    StructuredHistoryMemory,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RuleResult:
    """Outcome of a single symbolic rule evaluation."""

    rule_id: str
    rule_family: str  # 'workflow' | 'reachability' | 'leakage' | 'evidence' | 'discoverability'
    passed: bool
    severity: str  # 'high' | 'medium' | 'low'
    related_claim_ids: list[str] = field(default_factory=list)
    related_precondition_ids: list[str] = field(default_factory=list)
    evidence_unit_ids: list[int] = field(default_factory=list)
    reason: str = ''
    # When True, the symbolic check failed but the answer is ambiguous —
    # a focused LLM call should adjudicate before this rule is treated as
    # a hard pass/fail.  ``llm_context`` carries structured evidence for
    # the LLM prompt.
    needs_llm_assist: bool = False
    llm_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            'rule_id': self.rule_id,
            'rule_family': self.rule_family,
            'passed': self.passed,
            'severity': self.severity,
            'related_claim_ids': self.related_claim_ids,
            'related_precondition_ids': self.related_precondition_ids,
            'evidence_unit_ids': self.evidence_unit_ids,
            'reason': self.reason,
            'needs_llm_assist': self.needs_llm_assist,
        }
        return d


# Ordered phase sequence.  Index order encodes ordering constraints.
PHASE_ORDER: list[str] = [
    'reading',
    'running',
    'exploration',
    'test_creation',
    'fix_analysis',
    'fix_implementation',
    'verification',
    'final_review',
]

_PHASE_INDEX: dict[str, int] = {p: i for i, p in enumerate(PHASE_ORDER)}

# Minimum analysis length (chars) to count as "adequate Phase 5"
_MIN_ANALYSIS_LENGTH = 80

# Heuristic keywords to detect "analysis-like" content in think actions
_ANALYSIS_INDICATORS = re.compile(
    r'the (?:bug|issue|problem) (?:is|occurs|happens)|'
    r'root cause|'
    r'because|'
    r'the fix (?:is|should|would)|'
    r'needs? to (?:be )?(?:changed|fixed|replaced|updated)|'
    r'should (?:return|use|be|handle)',
    re.IGNORECASE,
)

# Heuristic keywords that indicate an *exploratory* action (read/view/search/find).
# Claims of type 'action' containing these are exempt from C3 oracle-dependence.
_EXPLORATORY_ACTION_RE = re.compile(
    r'\b(?:view|read|examine|inspect|look\s+at|open|cat|head|tail|less|'
    r'grep|find|search|locate|list|ls|tree|check|see|explore|discover|'
    r'navigate|investigate|trace|browse)\b',
    re.IGNORECASE,
)


def _path_suffixes(filepath: str) -> list[str]:
    """Return progressively shorter path suffixes for cross-prefix matching.

    For ``/workspace/django__django__3.1/django/db/models/lookups.py`` returns:
    - ``django__django__3.1/django/db/models/lookups.py``
    - ``django/db/models/lookups.py``
    - ``db/models/lookups.py``
    - ``models/lookups.py``
    - ``lookups.py``

    For matching purposes: if ANY suffix of a proposed path matches ANY suffix
    of a path in the issue text, the paths are considered equivalent.
    """
    parts = [p for p in filepath.replace('\\', '/').split('/') if p]
    # Build progressively shorter suffixes (skip the root-only component)
    suffixes: list[str] = []
    for i in range(len(parts)):
        suffix = '/'.join(parts[i:])
        if suffix:
            suffixes.append(suffix)
    return suffixes


def _path_in_text(filepath: str, text: str) -> bool:
    """Check if *filepath* (or a meaningful suffix) appears in *text*.

    Handles cross-workspace path prefix mismatches, e.g.:
    - proposal path: ``/workspace/django__django__3.1/django/db/models/lookups.py``
    - issue text path: ``/Users/u/.virtualenvs/.../django/db/models/lookups.py``

    Both share the suffix ``django/db/models/lookups.py``.
    """
    text_lower = text.lower()
    # Quick exact check
    if filepath.lower() in text_lower:
        return True
    # Check suffixes — require at least 2 path components to avoid false positives
    # on bare filenames like "models.py" which are too generic
    for suffix in _path_suffixes(filepath):
        if '/' in suffix and suffix.lower() in text_lower:
            return True
    # Also check basename for less common filenames (>10 chars or contains underscore)
    import os
    basename = os.path.basename(filepath)
    if basename and (len(basename) > 10 or '_' in basename) and basename.lower() in text_lower:
        return True
    return False


def _paths_match(path_a: str, path_b: str) -> bool:
    """Check if two paths refer to the same file despite different prefixes.

    Uses suffix intersection: if any multi-component suffix appears in both
    paths, they are considered equivalent.
    """
    if path_a == path_b:
        return True
    # Compare suffixes — require at least 2 components to avoid false positives
    suffixes_a = set(s.lower() for s in _path_suffixes(path_a) if '/' in s)
    suffixes_b = set(s.lower() for s in _path_suffixes(path_b) if '/' in s)
    return bool(suffixes_a & suffixes_b)


class SymbolicRuleEngine:
    """Evaluate deterministic verification rules over claims and evidence.

    Usage::

        engine = SymbolicRuleEngine(memory, issue_text)
        results = engine.evaluate_all(claims, preconditions, retrieved_units)
    """

    def __init__(
        self,
        memory: StructuredHistoryMemory,
        issue_text: str = '',
    ) -> None:
        self.memory = memory
        self.issue_text = issue_text
        self._issue_lower = issue_text.lower()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_all(
        self,
        claims: list[Claim],
        preconditions: list[Precondition],
        retrieved_units: list[HistoryUnit],
        fact_preconditions: list[dict] | None = None,
    ) -> list[RuleResult]:
        """Run all applicable rules and return their results."""
        results: list[RuleResult] = []

        # Family A — workflow
        results.extend(self._family_A(claims, preconditions))
        # Family B — reachability
        results.extend(self._family_B(claims, preconditions, retrieved_units))
        # Family C — leakage
        results.extend(self._family_C(claims, preconditions, retrieved_units))
        # Family D — evidence sufficiency
        results.extend(self._family_D(claims, preconditions, retrieved_units))
        # D4 — fact prerequisite visibility
        if fact_preconditions:
            results.append(self._rule_D4_fact_prerequisites_visible(fact_preconditions))
        # Family E — discoverability
        results.extend(self._family_E(claims, preconditions))

        return results

    def get_failed_high(self, results: list[RuleResult]) -> list[RuleResult]:
        """Filter for high-severity failures."""
        return [r for r in results if not r.passed and r.severity == 'high']

    def get_uncertain(self, results: list[RuleResult]) -> list[RuleResult]:
        """Filter for rules that need LLM assistance (C-family uncertain)."""
        return [r for r in results if r.rule_family == 'leakage' and not r.passed]

    @staticmethod
    def get_llm_assist_needed(results: list[RuleResult]) -> list[RuleResult]:
        """Filter for rules that failed symbolically but should be adjudicated by LLM."""
        return [r for r in results if r.needs_llm_assist and not r.passed]

    @staticmethod
    def get_failed_high_excluding_llm_assist(results: list[RuleResult]) -> list[RuleResult]:
        """High-severity failures that are NOT marked for LLM resolution.

        These are deterministic rejects (A-family, B1, B3) that don't need
        LLM judgment.
        """
        return [
            r for r in results
            if not r.passed and r.severity == 'high' and not r.needs_llm_assist
        ]

    # ------------------------------------------------------------------
    # Family A — Workflow phase ordering
    # ------------------------------------------------------------------

    def _family_A(
        self,
        claims: list[Claim],
        preconditions: list[Precondition],
    ) -> list[RuleResult]:
        results: list[RuleResult] = []

        edit_claims = [c for c in claims if c.claim_type == 'edit']
        workflow_claims = [c for c in claims if c.claim_type == 'workflow']
        workflow_pcs = [p for p in preconditions if p.category == 'workflow']

        # A-family rules (workflow ordering) have been removed.
        # They were template-specific and caused false rejections —
        # e.g. rejecting a runtime probe as "verification before implementation".
        # The planner prompt now provides workflow as a soft guideline.

        return results

    def _rule_A1_edit_requires_analysis(
        self,
        edit_claims: list[Claim],
        workflow_pcs: list[Precondition],
    ) -> RuleResult:
        """A1: Reject edit proposals if no Phase 5 analysis exists in history."""
        # Look for think actions with analysis content
        analysis_units = self.memory.phase_search('fix_analysis')
        has_adequate_analysis = False
        evidence_ids: list[int] = []

        for unit in analysis_units:
            if 'think' in unit.tags:
                text = unit.action_text
                if (
                    len(text) >= _MIN_ANALYSIS_LENGTH
                    and _ANALYSIS_INDICATORS.search(text)
                ):
                    has_adequate_analysis = True
                    evidence_ids.append(unit.unit_id)

        # Also check for analysis-like reasoning in agent messages
        if not has_adequate_analysis:
            think_units = self.memory.tag_search({'think'})
            for unit in think_units:
                text = unit.action_text
                if (
                    len(text) >= _MIN_ANALYSIS_LENGTH
                    and _ANALYSIS_INDICATORS.search(text)
                ):
                    has_adequate_analysis = True
                    evidence_ids.append(unit.unit_id)

        pc_ids = [p.precondition_id for p in workflow_pcs
                  if 'analysis' in p.text.lower() or 'phase 5' in p.text.lower()]

        return RuleResult(
            rule_id='workflow.A1_edit_requires_analysis',
            rule_family='workflow',
            passed=has_adequate_analysis,
            severity='high',
            related_claim_ids=[c.claim_id for c in edit_claims],
            related_precondition_ids=pc_ids,
            evidence_unit_ids=evidence_ids,
            reason=(
                'Fix analysis (Phase 5) evidence found in history.'
                if has_adequate_analysis
                else 'No adequate fix analysis (Phase 5) found in history. '
                     'A think action with root-cause analysis is required before editing.'
            ),
        )

    def _rule_A2_verification_requires_implementation(
        self,
        workflow_claims: list[Claim],
        workflow_pcs: list[Precondition],
    ) -> RuleResult:
        """A2: Reject verification proposals if no code edit exists in history."""
        has_edit = self.memory.has_edit_action()
        # Use list comprehension — tag_search returns list, not set
        edit_units_list = [
            u for u in self.memory.tag_search({'edit'})
            if 'test_edit' not in u.tags
        ] if has_edit else []
        evidence_ids = [u.unit_id for u in edit_units_list]

        return RuleResult(
            rule_id='workflow.A2_verification_requires_implementation',
            rule_family='workflow',
            passed=has_edit,
            severity='high',
            related_claim_ids=[c.claim_id for c in workflow_claims],
            related_precondition_ids=[],
            evidence_unit_ids=evidence_ids,
            reason=(
                'Code edit (Phase 6) found in history; verification is valid.'
                if has_edit
                else 'No code edit found in history. Verification (Phase 7) requires '
                     'implementation (Phase 6) first.'
            ),
        )

    def _rule_A3_finalization_requires_verification(
        self,
        workflow_claims: list[Claim],
        workflow_pcs: list[Precondition],
    ) -> RuleResult:
        """A3: Reject finalization proposals if no test run after edit."""
        has_test_after_edit = self.memory.has_test_run_after_edit()
        evidence_ids: list[int] = []
        if has_test_after_edit:
            # Find verification units
            for u in self.memory.tag_search({'test_run'}):
                evidence_ids.append(u.unit_id)

        return RuleResult(
            rule_id='workflow.A3_finalization_requires_verification',
            rule_family='workflow',
            passed=has_test_after_edit,
            severity='high',
            related_claim_ids=[c.claim_id for c in workflow_claims],
            related_precondition_ids=[],
            evidence_unit_ids=evidence_ids,
            reason=(
                'Verification (Phase 7) evidence found; finalization is valid.'
                if has_test_after_edit
                else 'No test run found after code edit. Finalization (Phase 8) requires '
                     'verification (Phase 7) first.'
            ),
        )

    def _rule_A4_phase_completion_requires_evidence(
        self,
        claim: Claim,
        workflow_pcs: list[Precondition],
    ) -> RuleResult:
        """A4: Phase completion assertions must have explicit evidence."""
        text_lower = claim.text.lower()
        target_phase: str | None = None
        for phase in PHASE_ORDER:
            if phase.replace('_', ' ') in text_lower or phase in text_lower:
                target_phase = phase
                break

        if target_phase is None:
            return RuleResult(
                rule_id='workflow.A4_phase_completion_requires_evidence',
                rule_family='workflow',
                passed=True,  # Can't determine phase, pass
                severity='medium',
                related_claim_ids=[claim.claim_id],
                reason='Could not identify target phase; skipping.',
            )

        phase_units = self.memory.phase_search(target_phase)
        has_evidence = len(phase_units) > 0

        return RuleResult(
            rule_id='workflow.A4_phase_completion_requires_evidence',
            rule_family='workflow',
            passed=has_evidence,
            severity='medium',
            related_claim_ids=[claim.claim_id],
            evidence_unit_ids=[u.unit_id for u in phase_units],
            reason=(
                f'Evidence for phase "{target_phase}" found in history.'
                if has_evidence
                else f'No evidence for phase "{target_phase}" found. '
                     f'Phase completion claim is unsupported.'
            ),
        )

    # ---- workflow helpers -------------------------------------------------

    @staticmethod
    def _claims_mention_verification(
        workflow_claims: list[Claim],
        all_claims: list[Claim],
    ) -> bool:
        # Strong signals: explicit testing/verification phase keywords
        strong_kws = ('run test', 'pytest', 'check fix', 'phase 7',
                      'verification phase', 'run the test', 'execute test')
        for c in workflow_claims + all_claims:
            text_lower = c.text.lower()
            if any(kw in text_lower for kw in strong_kws):
                return True

        # Weak signal: 'verif' substring — only trigger for workflow claims
        # or action claims that look like test execution, NOT for reasoning
        # or localization claims where "verified" means "confirmed/checked".
        weak_kws = ('verif',)
        for c in workflow_claims:
            text_lower = c.text.lower()
            if any(kw in text_lower for kw in weak_kws):
                return True
        # For non-workflow claims, require co-occurrence with test-like terms
        test_context = ('test', 'assert', 'expect', 'pass', 'fail', 'suite')
        for c in all_claims:
            if c.claim_type in ('reasoning', 'localization'):
                continue  # Skip — "verified" in analysis is exploratory
            text_lower = c.text.lower()
            if (any(kw in text_lower for kw in weak_kws)
                    and any(t in text_lower for t in test_context)):
                return True
        return False

    @staticmethod
    def _claims_mention_finalization(
        workflow_claims: list[Claim],
        all_claims: list[Claim],
    ) -> bool:
        final_kws = ('final', 'review', 'complet', 'done', 'finish')
        for c in workflow_claims + all_claims:
            text_lower = c.text.lower()
            if any(kw in text_lower for kw in final_kws):
                return True
        return False

    @staticmethod
    def _claims_assert_completion(claim: Claim) -> bool:
        completion_kws = ('complete', 'done', 'finished', 'ready')
        return any(kw in claim.text.lower() for kw in completion_kws)

    # ------------------------------------------------------------------
    # Family B — Reachability
    # ------------------------------------------------------------------

    def _family_B(
        self,
        claims: list[Claim],
        preconditions: list[Precondition],
        retrieved_units: list[HistoryUnit],
    ) -> list[RuleResult]:
        results: list[RuleResult] = []

        # Collect all file paths and symbols across claims
        claims_with_files = [c for c in claims if c.file_paths]
        claims_with_symbols = [c for c in claims if c.symbols]
        edit_claims = [c for c in claims if c.claim_type == 'edit']

        reachability_pcs = [p for p in preconditions if p.category == 'reachability']

        # B1: File path justification
        if claims_with_files:
            results.append(self._rule_B1_file_path_justification(
                claims_with_files, reachability_pcs, retrieved_units,
            ))

        # B2: Symbol justification
        if claims_with_symbols:
            results.append(self._rule_B2_symbol_justification(
                claims_with_symbols, reachability_pcs, retrieved_units,
            ))

        # B3: Edit target justification
        if edit_claims:
            results.append(self._rule_B3_edit_target_justification(
                edit_claims, reachability_pcs,
            ))

        # B4: Action parameter justification (line ranges, search terms)
        claims_with_params = [c for c in claims if c.action_parameters]
        if claims_with_params:
            results.append(self._rule_B4_action_parameter_justification(
                claims_with_params, reachability_pcs, retrieved_units,
            ))

        return results

    def _rule_B1_file_path_justification(
        self,
        claims: list[Claim],
        pcs: list[Precondition],
        retrieved_units: list[HistoryUnit],
    ) -> RuleResult:
        """B1: Every proposed file path must be justified (issue, history, or search).

        Exempt: file paths from exploratory action claims (read/view/grep/find)
        since the action itself IS the discovery step.
        """
        known_files = self.memory.get_all_known_files()
        unjustified: list[str] = []
        evidence_ids: list[int] = []

        # Collect paths that are from exploratory claims (exempt from justification)
        exploratory_paths: set[str] = set()
        for c in claims:
            if c.claim_type in ('action', 'localization') and _EXPLORATORY_ACTION_RE.search(c.text):
                exploratory_paths.update(c.file_paths)

        all_paths: set[str] = set()
        for c in claims:
            all_paths.update(c.file_paths)

        for fp in all_paths:
            # Exploratory actions don't need prior file path provenance —
            # reading/searching IS how you discover a file.
            if fp in exploratory_paths:
                continue
            # Check issue text (with path normalization for cross-prefix matching)
            if _path_in_text(fp, self.issue_text):
                continue
            # Check known files (with cross-prefix matching)
            if any(_paths_match(fp, kf) for kf in known_files):
                file_units = self.memory.file_path_search(fp)
                # Also try suffix-based search if exact search returns nothing
                if not file_units:
                    for suffix in _path_suffixes(fp):
                        if '/' in suffix:
                            file_units = self.memory.file_path_search(suffix)
                            if file_units:
                                break
                evidence_ids.extend(u.unit_id for u in file_units)
                continue
            # Check retrieved evidence (with path normalization)
            found_in_evidence = False
            for u in retrieved_units:
                if _path_in_text(fp, u.full_text):
                    evidence_ids.append(u.unit_id)
                    found_in_evidence = True
            if found_in_evidence:
                continue
            unjustified.append(fp)

        passed = len(unjustified) == 0
        pc_ids = [p.precondition_id for p in pcs
                  if any(fp in p.text for fp in unjustified)]

        return RuleResult(
            rule_id='reachability.B1_file_path_justification',
            rule_family='reachability',
            passed=passed,
            severity='high',
            related_claim_ids=[c.claim_id for c in claims],
            related_precondition_ids=pc_ids,
            evidence_unit_ids=list(set(evidence_ids)),
            reason=(
                'All proposed file paths are justified by issue, history, or search results.'
                if passed
                else f'Unjustified file paths: {unjustified}. '
                     f'These paths are not visible in issue text, history, or search results.'
            ),
        )

    def _rule_B2_symbol_justification(
        self,
        claims: list[Claim],
        pcs: list[Precondition],
        retrieved_units: list[HistoryUnit],
    ) -> RuleResult:
        """B2: Proposed symbols must appear in issue text or history."""
        all_symbols: set[str] = set()
        for c in claims:
            all_symbols.update(c.symbols)

        # Collect all symbols visible in history
        history_symbols: set[str] = set()
        for u in self.memory.units:
            history_symbols.update(u.symbols_mentioned)
        # Also from retrieved units
        for u in retrieved_units:
            history_symbols.update(u.symbols_mentioned)

        unjustified: list[str] = []
        for sym in all_symbols:
            sym_lower = sym.lower()
            # Check issue text (both exact and dotted-name components)
            if sym_lower in self._issue_lower:
                continue
            # For dotted symbols like "Lookup.__init__", also check parts
            sym_parts = sym.split('.')
            if len(sym_parts) > 1 and all(
                part.lower() in self._issue_lower for part in sym_parts if part.strip('_')
            ):
                continue
            # Check history symbols (case-insensitive)
            if any(sym_lower == hs.lower() for hs in history_symbols):
                continue
            # Check full text of all history units (catches symbols in output)
            if any(sym in u.full_text for u in self.memory.units):
                continue
            # Check full text of retrieved units
            if any(sym in u.full_text for u in retrieved_units):
                continue
            unjustified.append(sym)

        passed = len(unjustified) == 0

        # Symbols might be derivable/implied even if not literally present —
        # escalate to LLM when symbolic check fails.
        llm_assist = not passed
        llm_ctx: dict[str, Any] = {}
        if llm_assist:
            # Collect nearby evidence for LLM to judge
            nearby_units = []
            for sym in unjustified[:5]:
                kw_hits = self.memory.keyword_search(sym.split('.')[:1], top_k=3)
                for u in kw_hits:
                    nearby_units.append({
                        'unit_id': u.unit_id,
                        'summary': u.action_summary[:120],
                        'text_snippet': u.full_text[:300],
                    })
            llm_ctx = {
                'unjustified_symbols': unjustified,
                'question': (
                    'The following symbols were not found literally in the issue text '
                    'or interaction history. Could they be reasonably inferred or derived '
                    'from the context? Symbols that are standard library/framework names, '
                    'or that are strongly implied by the code the agent has already seen, '
                    'should be considered justified.'
                ),
                'nearby_evidence': nearby_units[:10],
            }

        return RuleResult(
            rule_id='reachability.B2_symbol_justification',
            rule_family='reachability',
            passed=passed,
            severity='medium',
            related_claim_ids=[c.claim_id for c in claims],
            needs_llm_assist=llm_assist,
            llm_context=llm_ctx,
            reason=(
                'All proposed symbols are justified.'
                if passed
                else f'Unjustified symbols: {unjustified}. '
                     f'Not visible in issue text or interaction history.'
            ),
        )

    def _rule_B3_edit_target_justification(
        self,
        edit_claims: list[Claim],
        pcs: list[Precondition],
    ) -> RuleResult:
        """B3: Edit targets must have been read or discovered via search."""
        read_files = set(self.memory.get_all_read_files())
        searched_files = set(self.memory.get_all_searched_files())
        known = read_files | searched_files

        unjustified_targets: list[str] = []
        evidence_ids: list[int] = []

        for claim in edit_claims:
            for fp in claim.file_paths:
                # Check if this file (or any suffix match) was read or found
                if any(_paths_match(fp, kf) for kf in known):
                    file_units = self.memory.file_path_search(fp)
                    if not file_units:
                        for suffix in _path_suffixes(fp):
                            if '/' in suffix:
                                file_units = self.memory.file_path_search(suffix)
                                if file_units:
                                    break
                    evidence_ids.extend(u.unit_id for u in file_units)
                elif _path_in_text(fp, self.issue_text):
                    pass  # mentioned in issue
                else:
                    unjustified_targets.append(fp)

        passed = len(unjustified_targets) == 0
        pc_ids = [p.precondition_id for p in pcs if 'edit target' in p.text.lower()]

        return RuleResult(
            rule_id='reachability.B3_edit_target_justification',
            rule_family='reachability',
            passed=passed,
            severity='high',
            related_claim_ids=[c.claim_id for c in edit_claims],
            related_precondition_ids=pc_ids,
            evidence_unit_ids=list(set(evidence_ids)),
            reason=(
                'Edit target files have been previously read or discovered.'
                if passed
                else f'Edit targets not previously inspected: {unjustified_targets}. '
                     f'Files must be read or discovered via search before editing.'
            ),
        )

    def _rule_B4_action_parameter_justification(
        self,
        claims: list[Claim],
        pcs: list[Precondition],
        retrieved_units: list[HistoryUnit],
    ) -> RuleResult:
        """B4: Action parameters (line ranges, search terms) must be inferrable.

        When a proposal specifies ``view_range: [746, 810]`` or greps for a
        specific term, those values must originate from prior exploration output
        (grep line numbers, tracebacks, prior view output, issue text).
        Otherwise they likely represent oracle-leaked knowledge.
        """
        # 1. Gather all history text
        all_text = self.issue_text
        for u in self.memory.units:
            all_text += '\n' + u.full_text

        # 2. Extract "context line numbers" from history — numbers that
        #    appeared in line-number-like contexts:
        #    - grep output: "  123:class Foo" (digits followed by colon/tab)
        #    - tracebacks: "line 123"
        #    - view output: "  123  def method():" (leading digits in cat -n)
        context_line_nums: set[int] = set()
        # grep / cat -n style: start-of-line digits followed by : or tab
        for m in re.finditer(r'(?:^|\n)\s*(\d+)(?:[:\t|])', all_text):
            try:
                n = int(m.group(1))
                if 1 <= n <= 100000:
                    context_line_nums.add(n)
            except ValueError:
                pass
        # "line X" references (tracebacks, comments)
        for m in re.finditer(r'\blines?\s+(\d+)', all_text, re.IGNORECASE):
            try:
                context_line_nums.add(int(m.group(1)))
            except ValueError:
                pass
        # GitHub URL line fragments: #L235, #L235-L236, #L235-236
        for m in re.finditer(r'#L(\d+)(?:-L?(\d+))?', all_text):
            try:
                context_line_nums.add(int(m.group(1)))
                if m.group(2):
                    context_line_nums.add(int(m.group(2)))
            except ValueError:
                pass
        # Diff hunk headers: @@ -233,5 +233,5 @@
        for m in re.finditer(r'@@\s*-?(\d+)', all_text):
            try:
                n = int(m.group(1))
                if 1 <= n <= 100000:
                    context_line_nums.add(n)
            except ValueError:
                pass

        all_text_lower = all_text.lower()

        # 3. Check each claim's action parameters
        unjustified: list[str] = []

        # Pre-compute: which file paths have been justified (known in history/issue)?
        known_files = self.memory.get_all_known_files()

        for claim in claims:
            ap = claim.action_parameters
            if not ap:
                continue

            # Determine if this claim is an exploratory read/view/search action.
            # For exploratory actions on files that are already known, line
            # numbers are exempt — reading IS how you discover the content at
            # specific lines.  Strict line justification only applies to edits.
            is_exploratory = (
                claim.claim_type in ('action', 'localization')
                and _EXPLORATORY_ACTION_RE.search(claim.text)
            )
            file_is_known = False
            if is_exploratory and claim.file_paths:
                file_is_known = any(
                    _path_in_text(fp, self.issue_text)
                    or any(_paths_match(fp, kf) for kf in known_files)
                    for fp in claim.file_paths
                )

            # --- Line numbers / view_range ---
            #
            # For view_range, the START line must be justified; the END line
            # is auto-justified if it's within a reasonable span (≤150 lines)
            # of the justified start — agents typically request a class/function
            # block of ~50–100 lines after finding the start via grep.
            #
            # Exception: exploratory read/view on a known file — the agent is
            # discovering content, so line numbers don't need prior provenance.
            vr = ap.get('view_range')
            vr_start_justified = False
            if vr and len(vr) == 2:
                start_ln, end_ln = vr

                # Exploratory reads on a known file are exempt
                if is_exploratory and file_is_known:
                    vr_start_justified = True
                else:
                    start_ok = start_ln < 20 or any(
                        abs(start_ln - cn) <= 10 for cn in context_line_nums
                    )
                    if start_ok:
                        vr_start_justified = True
                        # End line auto-justified if within reasonable span
                        if end_ln - start_ln > 150:
                            unjustified.append(f'view end line {end_ln} (>150 lines from start)')
                    else:
                        unjustified.append(f'view start line {start_ln}')
                        if end_ln >= 20 and not any(
                            abs(end_ln - cn) <= 10 for cn in context_line_nums
                        ):
                            unjustified.append(f'view end line {end_ln}')

            # Other standalone line numbers (not part of view_range)
            vr_set = set(vr) if vr else set()
            for ln in ap.get('line_numbers', []):
                if ln in vr_set:
                    continue  # Already handled above
                if ln < 20:
                    continue  # Small numbers are too common to be meaningful
                # Exploratory reads on a known file are exempt
                if is_exploratory and file_is_known:
                    continue
                # Justified if any context number is within ±10
                if any(abs(ln - cn) <= 10 for cn in context_line_nums):
                    continue
                unjustified.append(f'line {ln}')

            # --- Search terms (from grep commands) ---
            for term in ap.get('search_terms', []):
                if len(term) <= 2:
                    continue  # Too short to be meaningful
                term_lower = term.lower()
                if term_lower in all_text_lower:
                    continue
                # For compound patterns, check if all meaningful parts are present
                parts = [p for p in re.split(r'[_./\\|]+', term) if len(p) > 2]
                if parts and all(p.lower() in all_text_lower for p in parts):
                    continue
                # Agents compose grep patterns from Python keywords + known symbols
                # (e.g. "class Col", "def output_field"). Strip common keywords
                # and check if the remaining words appear in history.
                _grep_noise = {
                    'class', 'def', 'import', 'from', 'return', 'self',
                    'if', 'else', 'elif', 'for', 'while', 'with', 'as',
                    'try', 'except', 'finally', 'raise', 'yield', 'lambda',
                    'and', 'or', 'not', 'in', 'is', 'pass', 'break',
                    'continue', 'del', 'assert', 'global', 'nonlocal',
                }
                words = [w for w in re.split(r'[\s_./\\|]+', term)
                         if len(w) > 1 and w.lower() not in _grep_noise]
                if words and all(w.lower() in all_text_lower for w in words):
                    continue
                unjustified.append(f'search term "{term}"')

        passed = len(unjustified) == 0
        pc_ids = [
            p.precondition_id for p in pcs
            if 'line number' in p.text.lower() or 'search term' in p.text.lower()
        ]

        # Line numbers might be inferrable from broader context (e.g. a class
        # definition at line 220 makes viewing 225–250 reasonable).  Search terms
        # are well-handled symbolically, so only escalate when there are line
        # number failures.
        line_failures = [u for u in unjustified if 'line' in u]
        llm_assist = bool(line_failures)
        llm_ctx: dict[str, Any] = {}
        if llm_assist:
            # Provide the context line numbers and nearby evidence for LLM
            nearby_ctx = sorted(context_line_nums)[-20:] if context_line_nums else []
            llm_ctx = {
                'unjustified_params': unjustified,
                'question': (
                    'The following line numbers/ranges were not found within ±10 of '
                    'any line number in the interaction history. Could they be '
                    'reasonably inferred? For example, if a grep showed a class '
                    'starting at line 220, then viewing lines 225-250 is reasonable '
                    'even if 225 is not literally in the output. Check whether the '
                    'line numbers are plausible given the surrounding context.'
                ),
                'context_line_numbers_sample': nearby_ctx,
                'claim_texts': [c.text[:200] for c in claims if c.action_parameters],
            }

        return RuleResult(
            rule_id='reachability.B4_action_parameter_justification',
            rule_family='reachability',
            passed=passed,
            severity='high',
            related_claim_ids=[c.claim_id for c in claims],
            related_precondition_ids=pc_ids,
            needs_llm_assist=llm_assist,
            llm_context=llm_ctx,
            reason=(
                'All action parameters (line ranges, search terms) are justified '
                'by prior history.'
                if passed
                else f'Unjustified action parameters: {unjustified}. '
                     f'These values are not inferrable from issue text or '
                     f'interaction history. Use grep/search to discover line '
                     f'numbers before specifying view ranges.'
            ),
        )

    # ------------------------------------------------------------------
    # Family C — Leakage detection
    # ------------------------------------------------------------------

    def _family_C(
        self,
        claims: list[Claim],
        preconditions: list[Precondition],
        retrieved_units: list[HistoryUnit],
    ) -> list[RuleResult]:
        results: list[RuleResult] = []

        leakage_pcs = [p for p in preconditions if p.category == 'leakage']

        # C1: Hidden implementation detail
        results.append(self._rule_C1_hidden_implementation_detail(
            claims, leakage_pcs, retrieved_units,
        ))

        # C2: Unsupported localization
        localization_claims = [c for c in claims if c.claim_type == 'localization']
        if localization_claims:
            results.append(self._rule_C2_unsupported_localization(
                localization_claims, leakage_pcs, retrieved_units,
            ))

        # C3: Oracle-only dependence
        results.append(self._rule_C3_oracle_only_dependence(
            claims, leakage_pcs, retrieved_units,
        ))

        return results

    def _rule_C1_hidden_implementation_detail(
        self,
        claims: list[Claim],
        pcs: list[Precondition],
        retrieved_units: list[HistoryUnit],
    ) -> RuleResult:
        """C1: Reject if proposal introduces concrete code not in issue/history.

        This is a heuristic check — it looks for code-like patterns in claims
        that don't appear anywhere in the issue or history.  Complex cases
        may need LLM assist in Stage 4.
        """
        # Extract code-like snippets from claim texts
        code_snippets: list[str] = []
        for claim in claims:
            # Look for quoted code, backtick-wrapped code, or indented code
            snippets = re.findall(r'`([^`]+)`', claim.text)
            snippets += re.findall(r'"([^"]{5,})"', claim.text)
            code_snippets.extend(snippets)

        if not code_snippets:
            return RuleResult(
                rule_id='leakage.C1_hidden_implementation_detail',
                rule_family='leakage',
                passed=True,
                severity='high',
                related_claim_ids=[c.claim_id for c in claims],
                reason='No code-like snippets detected in claims.',
            )

        # Check each snippet against issue + history
        all_text = self._issue_lower
        for u in self.memory.units:
            all_text += '\n' + u.full_text.lower()
        for u in retrieved_units:
            all_text += '\n' + u.full_text.lower()

        unsupported: list[str] = []
        for snippet in code_snippets:
            if snippet.lower() not in all_text:
                unsupported.append(snippet)

        passed = len(unsupported) == 0

        # Code derivability is inherently subjective — a snippet might be a
        # trivial variation of code the agent has seen.  Escalate to LLM.
        llm_assist = not passed
        llm_ctx: dict[str, Any] = {}
        if llm_assist:
            llm_ctx = {
                'unsupported_snippets': unsupported[:5],
                'question': (
                    'The following code snippets from the proposal were not found '
                    'verbatim in the issue text or interaction history. Could they '
                    'be reasonably derived from code the agent has already seen? '
                    'Trivial variations (e.g. renaming, reordering, standard '
                    'patterns) should be considered supported.'
                ),
            }

        return RuleResult(
            rule_id='leakage.C1_hidden_implementation_detail',
            rule_family='leakage',
            passed=passed,
            severity='high',
            related_claim_ids=[c.claim_id for c in claims],
            related_precondition_ids=[p.precondition_id for p in pcs],
            needs_llm_assist=llm_assist,
            llm_context=llm_ctx,
            reason=(
                'All code snippets in proposal are grounded in issue or history.'
                if passed
                else f'Unsupported code snippets: {unsupported[:3]}. '
                     f'These do not appear in issue text or interaction history.'
            ),
        )

    def _rule_C2_unsupported_localization(
        self,
        localization_claims: list[Claim],
        pcs: list[Precondition],
        retrieved_units: list[HistoryUnit],
    ) -> RuleResult:
        """C2: Localization to file+method needs visible evidence chain.

        Exempt: localization claims that are part of an exploratory proposal
        (read/view/grep/search) — the localization IS the discovery step.
        """
        known_files = self.memory.get_all_known_files()
        history_lower = ''
        for u in self.memory.units:
            history_lower += u.full_text.lower() + '\n'
        for u in retrieved_units:
            history_lower += u.full_text.lower() + '\n'

        unsupported_localizations: list[str] = []
        for claim in localization_claims:
            # Exempt exploratory localization claims — if the claim describes
            # reading/viewing/searching, the localization is how the agent
            # discovers the file+method, not a leak.
            if _EXPLORATORY_ACTION_RE.search(claim.text):
                continue

            # Need both file and symbol to be supported
            files_ok = True
            if claim.file_paths:
                files_ok = all(
                    any(_paths_match(fp, kf) for kf in known_files)
                    or _path_in_text(fp, self.issue_text)
                    or _path_in_text(fp, history_lower)
                    for fp in claim.file_paths
                )

            symbols_ok = True
            if claim.symbols:
                symbols_ok = all(
                    sym.lower() in self._issue_lower
                    or sym.lower() in history_lower
                    # For dotted names, check if components are present
                    or (
                        '.' in sym and all(
                            part.lower() in self._issue_lower or part.lower() in history_lower
                            for part in sym.split('.') if part.strip('_')
                        )
                    )
                    for sym in claim.symbols
                )

            if not (files_ok and symbols_ok):
                unsupported_localizations.append(claim.text[:80])

        passed = len(unsupported_localizations) == 0

        # Whether a file+method combination is "established" can be subtle —
        # the agent might have seen the file but not the exact method name,
        # yet the method is inferrable.  Escalate to LLM.
        llm_assist = not passed
        llm_ctx: dict[str, Any] = {}
        if llm_assist:
            llm_ctx = {
                'unsupported_localizations': unsupported_localizations,
                'question': (
                    'The following localization claims reference file+method '
                    'combinations not literally established in the interaction '
                    'history. Could the agent have reasonably inferred these '
                    'localizations from the code it has already seen? For example, '
                    'if the agent read a file and saw a class definition, methods '
                    'of that class are inferrable.'
                ),
            }

        return RuleResult(
            rule_id='leakage.C2_unsupported_localization',
            rule_family='leakage',
            passed=passed,
            severity='high',
            related_claim_ids=[c.claim_id for c in localization_claims],
            related_precondition_ids=[p.precondition_id for p in pcs],
            needs_llm_assist=llm_assist,
            llm_context=llm_ctx,
            reason=(
                'All localization claims are supported by visible evidence.'
                if passed
                else f'Unsupported localizations: {unsupported_localizations}. '
                     f'File/method combination not established in history.'
            ),
        )

    def _rule_C3_oracle_only_dependence(
        self,
        claims: list[Claim],
        pcs: list[Precondition],
        retrieved_units: list[HistoryUnit],
    ) -> RuleResult:
        """C3: Flag claims that have no public provenance at all.

        A claim depends only on oracle if none of its file paths or symbols
        appear anywhere in issue text or history.

        Exempt: action claims that are purely exploratory (read/view/search)
        since the action itself IS the discovery step.
        """
        oracle_dependent_claims: list[str] = []

        for claim in claims:
            # Skip generic claims with no specific paths/symbols
            if not claim.file_paths and not claim.symbols:
                continue

            # Exempt exploratory action claims — reading/viewing/searching a file
            # IS how you discover it; the file doesn't need prior provenance.
            # This also covers localization claims that describe exploratory actions.
            if claim.claim_type in ('action', 'localization') and _EXPLORATORY_ACTION_RE.search(claim.text):
                continue

            has_public_support = False

            # Check files (with path normalization)
            for fp in claim.file_paths:
                if _path_in_text(fp, self.issue_text):
                    has_public_support = True
                    break
                if any(_path_in_text(fp, u.full_text) for u in self.memory.units):
                    has_public_support = True
                    break
                if any(_path_in_text(fp, u.full_text) for u in retrieved_units):
                    has_public_support = True
                    break

            if not has_public_support:
                # Check symbols
                for sym in claim.symbols:
                    sym_lower = sym.lower()
                    if sym_lower in self._issue_lower:
                        has_public_support = True
                        break
                    # Check dotted symbol parts
                    if '.' in sym:
                        parts = [p for p in sym.split('.') if p.strip('_')]
                        if parts and all(
                            p.lower() in self._issue_lower for p in parts
                        ):
                            has_public_support = True
                            break
                    if any(sym in u.full_text for u in self.memory.units):
                        has_public_support = True
                        break

            if not has_public_support:
                oracle_dependent_claims.append(claim.claim_id)

        passed = len(oracle_dependent_claims) == 0

        # Distinguishing inference from leakage requires judgment — the agent
        # might have seen related code that implies the claim.  Escalate.
        llm_assist = not passed
        llm_ctx: dict[str, Any] = {}
        if llm_assist:
            # Collect the claim texts for context
            flagged_texts = []
            for claim in claims:
                if claim.claim_id in oracle_dependent_claims:
                    flagged_texts.append({
                        'claim_id': claim.claim_id,
                        'text': claim.text[:200],
                        'file_paths': claim.file_paths,
                        'symbols': claim.symbols,
                    })
            llm_ctx = {
                'oracle_dependent_claims': flagged_texts,
                'question': (
                    'The following claims reference file paths or symbols that '
                    'were not found in the issue text or the agent\'s interaction '
                    'history. Could the agent have reasonably inferred this '
                    'knowledge from what it has already seen? Consider whether '
                    'the referenced files/symbols are mentioned in tracebacks, '
                    'imports, class hierarchies, or other indirect evidence.'
                ),
            }

        return RuleResult(
            rule_id='leakage.C3_oracle_only_dependence',
            rule_family='leakage',
            passed=passed,
            severity='high',
            related_claim_ids=oracle_dependent_claims,
            related_precondition_ids=[p.precondition_id for p in pcs],
            needs_llm_assist=llm_assist,
            llm_context=llm_ctx,
            reason=(
                'All claims with specific paths/symbols have public provenance.'
                if passed
                else f'Claims with no public provenance: {oracle_dependent_claims}. '
                     f'These appear to depend solely on oracle context.'
            ),
        )

    # ------------------------------------------------------------------
    # Family D — Evidence sufficiency
    # ------------------------------------------------------------------

    def _family_D(
        self,
        claims: list[Claim],
        preconditions: list[Precondition],
        retrieved_units: list[HistoryUnit],
    ) -> list[RuleResult]:
        results: list[RuleResult] = []

        evidence_pcs = [p for p in preconditions if p.category == 'evidence']

        # D1: Bug-cause support
        reasoning_claims = [c for c in claims if c.claim_type == 'reasoning']
        if reasoning_claims:
            results.append(self._rule_D1_bug_cause_support(
                reasoning_claims, evidence_pcs, retrieved_units,
            ))

        # D2: Test claim support
        test_claims = [c for c in claims
                       if re.search(r'\btest\b', c.text, re.IGNORECASE)]
        if test_claims:
            results.append(self._rule_D2_test_claim_support(test_claims, evidence_pcs))

        # D3: Analysis claim support
        analysis_claims = [c for c in claims if c.claim_type == 'reasoning']
        if analysis_claims:
            results.append(self._rule_D3_analysis_claim_support(
                analysis_claims, evidence_pcs,
            ))

        return results

    def _rule_D1_bug_cause_support(
        self,
        reasoning_claims: list[Claim],
        pcs: list[Precondition],
        retrieved_units: list[HistoryUnit],
    ) -> RuleResult:
        """D1: Bug-cause claims require at least one supporting evidence span."""
        # A reasoning claim is supported if any of its files/symbols appear
        # in observations (meaning the agent has actually looked at evidence).
        evidence_ids: list[int] = []
        has_support = False

        for claim in reasoning_claims:
            for fp in claim.file_paths:
                units = self.memory.file_path_search(fp)
                if units:
                    evidence_ids.extend(u.unit_id for u in units)
                    has_support = True

            for sym in claim.symbols:
                kw_results = self.memory.keyword_search([sym], top_k=3)
                if kw_results:
                    evidence_ids.extend(u.unit_id for u in kw_results)
                    has_support = True

        # Also check issue text as support
        if not has_support:
            for claim in reasoning_claims:
                if any(fp.lower() in self._issue_lower for fp in claim.file_paths):
                    has_support = True
                if any(sym.lower() in self._issue_lower for sym in claim.symbols):
                    has_support = True

        pc_ids = [p.precondition_id for p in pcs if 'bug' in p.text.lower() or 'cause' in p.text.lower()]

        # Whether evidence "supports" a reasoning claim is inherently abstract —
        # the agent may have seen related code even if the exact file/symbol
        # doesn't match.  Escalate to LLM for judgment.
        llm_assist = not has_support
        llm_ctx: dict[str, Any] = {}
        if llm_assist:
            llm_ctx = {
                'reasoning_claims': [c.text[:200] for c in reasoning_claims],
                'question': (
                    'The following bug-cause reasoning claims have no explicit '
                    'supporting evidence (no matching files or symbols in history). '
                    'Could the reasoning be supported by the general context the '
                    'agent has explored? Consider whether the agent has seen '
                    'enough code/errors to plausibly make these claims.'
                ),
            }

        return RuleResult(
            rule_id='evidence.D1_bug_cause_support',
            rule_family='evidence',
            passed=has_support,
            severity='medium',
            related_claim_ids=[c.claim_id for c in reasoning_claims],
            related_precondition_ids=pc_ids,
            evidence_unit_ids=list(set(evidence_ids)),
            needs_llm_assist=llm_assist,
            llm_context=llm_ctx,
            reason=(
                'Bug-cause reasoning has supporting evidence in history.'
                if has_support
                else 'Bug-cause reasoning has no supporting evidence. '
                     'The agent must observe relevant code or error output first.'
            ),
        )

    def _rule_D2_test_claim_support(
        self,
        test_claims: list[Claim],
        pcs: list[Precondition],
    ) -> RuleResult:
        """D2: Test-related claims require observed test/reproduction evidence."""
        test_units = self.memory.tag_search({'test_run'})
        test_edit_units = self.memory.tag_search({'test_edit'})
        repro_units = self.memory.keyword_search(
            ['reproduce', 'reproduction', 'repro'], top_k=5,
        )

        has_test_evidence = bool(test_units) or bool(test_edit_units) or bool(repro_units)
        evidence_ids = (
            [u.unit_id for u in test_units]
            + [u.unit_id for u in test_edit_units]
            + [u.unit_id for u in repro_units]
        )

        return RuleResult(
            rule_id='evidence.D2_test_claim_support',
            rule_family='evidence',
            passed=has_test_evidence,
            severity='medium',
            related_claim_ids=[c.claim_id for c in test_claims],
            evidence_unit_ids=list(set(evidence_ids)),
            reason=(
                'Test-related evidence found in history.'
                if has_test_evidence
                else 'No test or reproduction evidence found in history.'
            ),
        )

    def _rule_D3_analysis_claim_support(
        self,
        analysis_claims: list[Claim],
        pcs: list[Precondition],
    ) -> RuleResult:
        """D3: Analysis claims require explicit reasoning evidence."""
        think_units = self.memory.tag_search({'think'})
        # Also check exploration units that contain analysis-like content
        exploration_units = self.memory.phase_search('exploration')
        analysis_evidence = list(think_units)
        for u in exploration_units:
            if _ANALYSIS_INDICATORS.search(u.full_text):
                analysis_evidence.append(u)

        has_analysis = bool(analysis_evidence)

        return RuleResult(
            rule_id='evidence.D3_analysis_claim_support',
            rule_family='evidence',
            passed=has_analysis,
            severity='medium',
            related_claim_ids=[c.claim_id for c in analysis_claims],
            evidence_unit_ids=[u.unit_id for u in analysis_evidence],
            reason=(
                'Analysis/reasoning evidence found in history (think actions or analysis steps).'
                if has_analysis
                else 'No reasoning/analysis evidence found in history. '
                     'Fix-analysis claims require prior think actions or detailed reasoning steps.'
            ),
        )

    def _rule_D4_fact_prerequisites_visible(
        self,
        fact_preconditions: list[dict],
    ) -> RuleResult:
        """D4: Check that prerequisite facts' knowledge is visible in history.

        When the oracle planner references a fact that has prerequisites,
        those prerequisites' statements should be visible in the interaction
        history (the debugger should have already discovered that knowledge).
        If not, the proposal introduces an unjustified knowledge jump.

        This rule uses LLM assistance because checking whether a fact's
        statement is "visible" requires semantic matching, not just keyword search.
        """
        # Gather all history text for searching
        all_text = self.issue_text
        for u in self.memory.units:
            all_text += '\n' + u.full_text
        all_text_lower = all_text.lower()

        missing_prereqs: list[str] = []
        checked_prereqs: list[str] = []

        for fp in fact_preconditions:
            for pc_text in fp.get('preconditions', []):
                # pc_text is like "[f8] (fact): The function is_supported_format..."
                checked_prereqs.append(pc_text[:100])

                # Extract the statement part (after the first "): ")
                statement_part = pc_text
                if '): ' in pc_text:
                    statement_part = pc_text.split('): ', 1)[1]

                # Check: are the key technical terms from this prerequisite
                # visible in the history? We look for significant words.
                words = [w for w in re.split(r'[\s,;.()]+', statement_part)
                         if len(w) > 4 and w.lower() not in {
                             'which', 'where', 'there', 'their', 'these',
                             'those', 'about', 'would', 'could', 'should',
                             'function', 'returns', 'using'
                         }]

                if not words:
                    continue

                # Require at least 40% of significant words to appear
                found = sum(1 for w in words if w.lower() in all_text_lower)
                if len(words) > 0 and found < len(words) * 0.4:
                    missing_prereqs.append(pc_text[:150])

        passed = len(missing_prereqs) == 0

        # Use LLM assist for borderline cases
        llm_assist = bool(missing_prereqs)
        llm_ctx: dict = {}
        if llm_assist:
            llm_ctx = {
                'question': (
                    'The proposal relies on investigation facts whose prerequisites '
                    'may not be visible in the interaction history. Check whether '
                    'the knowledge described in each missing prerequisite has actually '
                    'been discovered/observed by the debugger. If the knowledge is '
                    'present even in a different form, the prerequisite is satisfied.'
                ),
                'missing_prerequisites': missing_prereqs,
                'checked_prerequisites': checked_prereqs,
            }

        return RuleResult(
            rule_id='evidence.D4_fact_prerequisites_visible',
            rule_family='evidence',
            passed=passed,
            severity='medium',
            related_claim_ids=[],
            needs_llm_assist=llm_assist,
            llm_context=llm_ctx,
            reason=(
                'All fact prerequisites are visible in the interaction history.'
                if passed
                else f'Missing prerequisite knowledge in history: {missing_prereqs}. '
                     'The proposal may rely on knowledge the debugger has not yet discovered.'
            ),
        )

    # ------------------------------------------------------------------
    # Family E — Discoverability (advisory, low severity)
    # ------------------------------------------------------------------

    def _family_E(
        self,
        claims: list[Claim],
        preconditions: list[Precondition],
    ) -> list[RuleResult]:
        results: list[RuleResult] = []

        edit_claims = [c for c in claims if c.claim_type == 'edit']

        # E1: Discoverable next step
        if edit_claims:
            results.append(self._rule_E1_discoverable_next_step(edit_claims))

        # E2: Missing prerequisite redirect
        # This runs if any *inferred* reachability precondition exists
        inferred_reach = [
            p for p in preconditions
            if p.source == 'inferred' and p.category == 'reachability'
        ]
        if inferred_reach:
            results.append(self._rule_E2_missing_prerequisite_redirect(
                claims, inferred_reach,
            ))

        return results

    def _rule_E1_discoverable_next_step(
        self,
        edit_claims: list[Claim],
    ) -> RuleResult:
        """E1: Prefer proposals that inspect before editing."""
        # Advisory: if the proposal edits a file, check whether that file
        # was recently read (within the last few units).
        read_files = set(self.memory.get_all_read_files())
        edit_paths: set[str] = set()
        for c in edit_claims:
            edit_paths.update(c.file_paths)

        not_recently_read: list[str] = []
        for fp in edit_paths:
            if not any(fp in rf or rf.endswith(fp) or fp.endswith(rf) for rf in read_files):
                not_recently_read.append(fp)

        safe = len(not_recently_read) == 0

        return RuleResult(
            rule_id='discoverability.E1_discoverable_next_step',
            rule_family='discoverability',
            passed=safe,
            severity='low',
            related_claim_ids=[c.claim_id for c in edit_claims],
            reason=(
                'Edit targets have been previously inspected.'
                if safe
                else f'Advisory: Consider inspecting {not_recently_read} before editing. '
                     f'Reading the file first produces better SFT data.'
            ),
        )

    def _rule_E2_missing_prerequisite_redirect(
        self,
        claims: list[Claim],
        inferred_pcs: list[Precondition],
    ) -> RuleResult:
        """E2: If evidence is insufficient, suggest the safest next discovery step."""
        suggestions: list[str] = []

        for pc in inferred_pcs:
            text_lower = pc.text.lower()
            if 'file' in text_lower and ('visible' in text_lower or 'discoverable' in text_lower):
                suggestions.append('grep or find to discover the file path')
            elif 'symbol' in text_lower:
                suggestions.append('grep for the symbol name in the codebase')
            elif 'inspect' in text_lower or 'read' in text_lower:
                suggestions.append('read the target file before editing')
            elif 'analysis' in text_lower:
                suggestions.append('use think tool to perform explicit fix analysis')

        # Deduplicate
        seen: set[str] = set()
        unique_suggestions: list[str] = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)

        return RuleResult(
            rule_id='discoverability.E2_missing_prerequisite_redirect',
            rule_family='discoverability',
            passed=len(unique_suggestions) == 0,
            severity='low',
            related_claim_ids=[c.claim_id for c in claims],
            related_precondition_ids=[p.precondition_id for p in inferred_pcs],
            reason=(
                'No missing prerequisite redirections needed.'
                if not unique_suggestions
                else 'Suggested safer next steps: ' + '; '.join(unique_suggestions)
            ),
        )
