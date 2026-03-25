"""Claim and precondition extraction for the neuro-symbolic verifier.

This module defines the structured types used throughout the verifier pipeline
and provides both an LLM-assisted ``ClaimExtractor`` and a programmatic fallback
``ProgrammaticClaimExtractor``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    """A single verifiable claim extracted from a planner proposal."""

    claim_id: str  # "c1", "c2", ...
    claim_type: str  # 'action' | 'reasoning' | 'workflow' | 'localization' | 'edit'
    text: str
    file_paths: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    action_parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            'claim_id': self.claim_id,
            'claim_type': self.claim_type,
            'text': self.text,
            'file_paths': self.file_paths,
            'symbols': self.symbols,
        }
        if self.action_parameters:
            d['action_parameters'] = self.action_parameters
        return d


@dataclass
class Precondition:
    """A precondition that must be satisfied before a proposal is valid."""

    precondition_id: str  # "p1", "p2", ...
    source: str  # 'oracle_json' | 'inferred'
    text: str
    category: str  # 'workflow' | 'reachability' | 'evidence' | 'leakage'

    def to_dict(self) -> dict[str, Any]:
        return {
            'precondition_id': self.precondition_id,
            'source': self.source,
            'text': self.text,
            'category': self.category,
        }


@dataclass
class ProofObligation:
    """A proof obligation linking a claim to a precondition."""

    obligation_id: str  # "o1", "o2", ...
    claim_id: str
    precondition_id: str | None
    description: str
    retrieval_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'obligation_id': self.obligation_id,
            'claim_id': self.claim_id,
            'precondition_id': self.precondition_id,
            'description': self.description,
            'retrieval_hints': self.retrieval_hints,
        }


@dataclass
class ExtractionResult:
    """Complete output of claim/precondition extraction from a proposal."""

    claims: list[Claim] = field(default_factory=list)
    explicit_preconditions: list[Precondition] = field(default_factory=list)
    inferred_preconditions: list[Precondition] = field(default_factory=list)
    proof_obligations: list[ProofObligation] = field(default_factory=list)
    retrieval_plan: list[str] = field(default_factory=list)
    raw_llm_response: str = ''
    raw_llm_prompt: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'claims': [c.to_dict() for c in self.claims],
            'explicit_preconditions': [p.to_dict() for p in self.explicit_preconditions],
            'inferred_preconditions': [p.to_dict() for p in self.inferred_preconditions],
            'proof_obligations': [o.to_dict() for o in self.proof_obligations],
            'retrieval_plan': self.retrieval_plan,
        }

    @property
    def all_preconditions(self) -> list[Precondition]:
        return self.explicit_preconditions + self.inferred_preconditions


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_FILE_PATH_RE = re.compile(r'(?:/[\w._-]+){2,}(?:\.\w+)?')
_TOOL_CALL_RE = re.compile(r'\[TOOL CALL\]\s*(\w+)\(', re.IGNORECASE)
_STR_REPLACE_RE = re.compile(r'str_replace|str_replace_editor', re.IGNORECASE)
_EDIT_KEYWORDS_RE = re.compile(
    r'\b(?:edit|replace|patch|modify|change|fix|update)\b.*\b(?:file|code|line|function|method)\b',
    re.IGNORECASE,
)
_ANALYSIS_KEYWORDS_RE = re.compile(
    r'\b(?:the bug is|the issue is|root cause|because|the problem is|should be|needs to)\b',
    re.IGNORECASE,
)
_WORKFLOW_KEYWORDS_RE = re.compile(
    r'\b(?:phase \d|move to|proceed to|verification|final review|analysis complete)\b',
    re.IGNORECASE,
)
_PYTHON_SYMBOL_RE = re.compile(r'\b(?:class|def)\s+(\w+)|\b(\w+(?:\.\w+)+)\s*\(')
_TEST_RE = re.compile(r'\b(?:test|pytest|unittest|reproduce|reproduction)\b', re.IGNORECASE)
_LOCALIZATION_RE = re.compile(
    r'\b(?:in file|in method|at line|function|class)\s+[`"]?[\w./]+',
    re.IGNORECASE,
)

# --- Action parameter extraction regexes ---
_LINE_REF_RE = re.compile(
    r'lines?\s+(\d+)\s*[-–to]+\s*(\d+)'      # "lines 746-810", "line 746 to 810"
    r'|\bline\s+(\d+)\b',                      # "line 746"
    re.IGNORECASE,
)
_VIEW_RANGE_RE = re.compile(r'"view_range"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]')


def _extract_tool_call_args(text: str) -> dict[str, Any] | None:
    """Extract the JSON arguments dict from a ``[TOOL CALL]`` in *text*."""
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    # Find the opening brace after the tool name
    start = m.end()
    brace_start = text.find('{', start)
    if brace_start == -1:
        return None
    # Find matching closing brace via depth counting
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                json_str = text[brace_start:i + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return None
    return None


def _extract_grep_terms(cmd: str) -> list[str]:
    """Extract search patterns from a bash grep/find command string."""
    terms: list[str] = []
    # Quoted patterns
    for m in re.finditer(r"grep\s+(?:-[^\s]+\s+)*'([^']+)'", cmd):
        terms.append(m.group(1))
    for m in re.finditer(r'grep\s+(?:-[^\s]+\s+)*"([^"]+)"', cmd):
        terms.append(m.group(1))
    if not terms:
        # Unquoted single-word pattern (after flags)
        m = re.search(r'grep\s+(?:-\S+\s+)*(\w[\w.*+?-]+)', cmd)
        if m:
            term = m.group(1)
            # Skip common flags that look like words
            if term not in ('r', 'n', 'l', 'i', 'c', 'v', 'w', 'rn', 'rni'):
                terms.append(term)
    return terms


def _extract_action_params(proposal_text: str) -> dict[str, Any]:
    """Extract action-specific parameters from tool calls in a proposal.

    Extracts:
    - ``view_range`` / ``line_numbers``: specific line ranges the proposal targets
    - ``search_terms``: grep/search patterns from bash commands
    - ``old_str``: the original code string in a str_replace operation
    """
    params: dict[str, Any] = {}
    line_nums: set[int] = set()

    # 1. Parse [TOOL CALL] JSON
    tool_args = _extract_tool_call_args(proposal_text)
    if tool_args:
        # view_range
        vr = tool_args.get('view_range')
        if isinstance(vr, list) and len(vr) >= 2:
            try:
                params['view_range'] = [int(vr[0]), int(vr[1])]
                line_nums.update(params['view_range'])
            except (ValueError, TypeError):
                pass

        # bash command → grep terms
        cmd = tool_args.get('command', '')
        if isinstance(cmd, str) and re.search(r'\bgrep\b', cmd, re.IGNORECASE):
            terms = _extract_grep_terms(cmd)
            if terms:
                params['search_terms'] = terms

        # str_replace old_str (for edit parameter checking)
        old_str = tool_args.get('old_str')
        if isinstance(old_str, str) and len(old_str) > 10:
            params['old_str'] = old_str

    # 2. Extract explicit line number references from proposal reasoning text
    for m in _LINE_REF_RE.finditer(proposal_text):
        if m.group(1) and m.group(2):
            line_nums.add(int(m.group(1)))
            line_nums.add(int(m.group(2)))
        elif m.group(3):
            line_nums.add(int(m.group(3)))

    # 3. Extract from view_range JSON pattern in raw text (backup)
    for m in _VIEW_RANGE_RE.finditer(proposal_text):
        line_nums.add(int(m.group(1)))
        line_nums.add(int(m.group(2)))

    if line_nums:
        params['line_numbers'] = sorted(line_nums)

    return params


# ---------------------------------------------------------------------------
# Programmatic (fallback) extractor
# ---------------------------------------------------------------------------


class ProgrammaticClaimExtractor:
    """Rule-based claim/precondition extractor (no LLM).

    This serves as:
    1. The fallback when the LLM extractor fails.
    2. The standalone extractor until the LLM version is implemented in Phase 2.

    It uses regex and keyword heuristics to decompose a proposal into
    structured claims and infer preconditions.
    """

    def extract(
        self,
        proposal_text: str,
        oracle_preconditions: list[dict] | None = None,
        issue_text: str = '',
    ) -> ExtractionResult:
        claims = self._extract_claims(proposal_text)
        explicit = self._build_explicit_preconditions(oracle_preconditions)
        inferred = self._infer_preconditions(claims, proposal_text, issue_text)
        obligations = self._build_obligations(claims, explicit + inferred)
        retrieval_plan = self._build_retrieval_plan(claims, inferred)

        return ExtractionResult(
            claims=claims,
            explicit_preconditions=explicit,
            inferred_preconditions=inferred,
            proof_obligations=obligations,
            retrieval_plan=retrieval_plan,
        )

    # ---- claim extraction ------------------------------------------------

    def _extract_claims(self, proposal_text: str) -> list[Claim]:
        claims: list[Claim] = []
        claim_idx = 0

        # --- Extract file paths mentioned in proposal ---
        file_paths = _FILE_PATH_RE.findall(proposal_text)

        # --- Extract symbols ---
        symbols: list[str] = []
        for m in _PYTHON_SYMBOL_RE.finditer(proposal_text):
            sym = m.group(1) or m.group(2)
            if sym and sym not in symbols:
                symbols.append(sym)

        # --- Check for tool call ---
        tool_match = _TOOL_CALL_RE.search(proposal_text)
        tool_name = tool_match.group(1) if tool_match else None

        # --- Determine claim types ---

        # Edit claim
        if tool_name and _STR_REPLACE_RE.match(tool_name):
            claim_idx += 1
            claims.append(Claim(
                claim_id=f'c{claim_idx}',
                claim_type='edit',
                text=f'Proposal suggests editing via {tool_name}',
                file_paths=file_paths[:],
                symbols=symbols[:],
            ))
        elif _EDIT_KEYWORDS_RE.search(proposal_text):
            claim_idx += 1
            claims.append(Claim(
                claim_id=f'c{claim_idx}',
                claim_type='edit',
                text='Proposal suggests code modification',
                file_paths=file_paths[:],
                symbols=symbols[:],
            ))

        # Action claim (non-edit tool call)
        if tool_name and not _STR_REPLACE_RE.match(tool_name):
            claim_idx += 1
            claims.append(Claim(
                claim_id=f'c{claim_idx}',
                claim_type='action',
                text=f'Proposal suggests action via {tool_name}',
                file_paths=file_paths[:],
                symbols=symbols[:],
            ))

        # Reasoning / analysis claim
        if _ANALYSIS_KEYWORDS_RE.search(proposal_text):
            claim_idx += 1
            claims.append(Claim(
                claim_id=f'c{claim_idx}',
                claim_type='reasoning',
                text='Proposal contains reasoning about bug cause or fix',
                file_paths=file_paths[:],
                symbols=symbols[:],
            ))

        # Workflow claim
        if _WORKFLOW_KEYWORDS_RE.search(proposal_text):
            claim_idx += 1
            claims.append(Claim(
                claim_id=f'c{claim_idx}',
                claim_type='workflow',
                text='Proposal references workflow phase transition',
                file_paths=file_paths[:],
                symbols=symbols[:],
            ))

        # Localization claim
        if _LOCALIZATION_RE.search(proposal_text):
            claim_idx += 1
            claims.append(Claim(
                claim_id=f'c{claim_idx}',
                claim_type='localization',
                text='Proposal localizes issue to specific file/method/line',
                file_paths=file_paths[:],
                symbols=symbols[:],
            ))

        # Fallback: if no specific claims detected, create a generic one
        if not claims:
            claim_idx += 1
            claims.append(Claim(
                claim_id=f'c{claim_idx}',
                claim_type='action',
                text='Generic proposal action',
                file_paths=file_paths[:],
                symbols=symbols[:],
            ))

        # Enrich claims with parsed action parameters (line ranges, search terms)
        action_params = _extract_action_params(proposal_text)
        if action_params:
            for claim in claims:
                if claim.claim_type in ('action', 'localization', 'edit'):
                    claim.action_parameters = action_params

        return claims

    # ---- precondition extraction -----------------------------------------

    @staticmethod
    def _build_explicit_preconditions(oracle_preconditions: list[dict] | None) -> list[Precondition]:
        if not oracle_preconditions:
            return []
        result: list[Precondition] = []
        pid = 0
        for fp in oracle_preconditions:
            fact_id = fp.get('fact_id', '')
            for pc_text in fp.get('preconditions', []):
                pid += 1
                result.append(Precondition(
                    precondition_id=f'p{pid}',
                    source='oracle_json',
                    text=f'[{fact_id}] {pc_text}' if fact_id else pc_text,
                    category='evidence',
                ))
        return result

    @staticmethod
    def _infer_preconditions(
        claims: list[Claim],
        proposal_text: str,
        issue_text: str,
    ) -> list[Precondition]:
        inferred: list[Precondition] = []
        pid = 100  # start at 100 to avoid collision with explicit

        for claim in claims:
            # File paths in claim → reachability precondition
            for fp in claim.file_paths:
                if fp.lower() not in issue_text.lower():
                    pid += 1
                    inferred.append(Precondition(
                        precondition_id=f'p{pid}',
                        source='inferred',
                        text=f'File path {fp} must be visible in history or discoverable',
                        category='reachability',
                    ))

            # Symbols in claim → reachability precondition
            for sym in claim.symbols:
                if sym.lower() not in issue_text.lower():
                    pid += 1
                    inferred.append(Precondition(
                        precondition_id=f'p{pid}',
                        source='inferred',
                        text=f'Symbol {sym} must be visible in issue or history',
                        category='reachability',
                    ))

            # Edit claim → must have inspected the file
            if claim.claim_type == 'edit':
                pid += 1
                inferred.append(Precondition(
                    precondition_id=f'p{pid}',
                    source='inferred',
                    text='Edit target file must have been read/inspected in prior history',
                    category='reachability',
                ))
                pid += 1
                inferred.append(Precondition(
                    precondition_id=f'p{pid}',
                    source='inferred',
                    text='Fix analysis (Phase 5) must be completed before edit (Phase 6)',
                    category='workflow',
                ))

            # Reasoning claim → must have evidence
            if claim.claim_type == 'reasoning':
                pid += 1
                inferred.append(Precondition(
                    precondition_id=f'p{pid}',
                    source='inferred',
                    text='Bug-cause reasoning must be supported by observed evidence',
                    category='evidence',
                ))

            # Localization claim → evidence chain required
            if claim.claim_type == 'localization':
                pid += 1
                inferred.append(Precondition(
                    precondition_id=f'p{pid}',
                    source='inferred',
                    text='Localization to specific file/method must have visible evidence chain',
                    category='leakage',
                ))

            # Action parameters → parameter justification preconditions
            if claim.action_parameters:
                ap = claim.action_parameters
                if 'line_numbers' in ap:
                    line_nums = ap['line_numbers']
                    pid += 1
                    inferred.append(Precondition(
                        precondition_id=f'p{pid}',
                        source='inferred',
                        text=f'Line numbers {line_nums} must be inferrable from prior exploration output (grep, traceback, or prior view)',
                        category='reachability',
                    ))
                if 'search_terms' in ap:
                    terms = ap['search_terms']
                    pid += 1
                    inferred.append(Precondition(
                        precondition_id=f'p{pid}',
                        source='inferred',
                        text=f'Search terms {terms} must appear in issue text or interaction history',
                        category='reachability',
                    ))

        # Deduplicate by text
        seen_texts: set[str] = set()
        deduped: list[Precondition] = []
        for p in inferred:
            if p.text not in seen_texts:
                seen_texts.add(p.text)
                deduped.append(p)

        return deduped

    # ---- proof obligations -----------------------------------------------

    @staticmethod
    def _build_obligations(
        claims: list[Claim],
        preconditions: list[Precondition],
    ) -> list[ProofObligation]:
        obligations: list[ProofObligation] = []
        oid = 0
        for claim in claims:
            for pc in preconditions:
                # Link obligation if precondition category matches claim type
                if _obligation_matches(claim, pc):
                    oid += 1
                    obligations.append(ProofObligation(
                        obligation_id=f'o{oid}',
                        claim_id=claim.claim_id,
                        precondition_id=pc.precondition_id,
                        description=f'Verify: {pc.text} (for claim: {claim.text[:80]})',
                        retrieval_hints=claim.file_paths[:3] + claim.symbols[:3],
                    ))
        return obligations

    # ---- retrieval plan --------------------------------------------------

    @staticmethod
    def _build_retrieval_plan(
        claims: list[Claim],
        inferred: list[Precondition],
    ) -> list[str]:
        plan: list[str] = []
        seen: set[str] = set()

        # File path queries
        for claim in claims:
            for fp in claim.file_paths:
                query = f'file:{fp}'
                if query not in seen:
                    seen.add(query)
                    plan.append(query)

        # Symbol queries
        for claim in claims:
            for sym in claim.symbols:
                query = f'symbol:{sym}'
                if query not in seen:
                    seen.add(query)
                    plan.append(query)

        # Phase queries based on inferred preconditions
        for pc in inferred:
            if pc.category == 'workflow':
                if 'analysis' in pc.text.lower() or 'phase 5' in pc.text.lower():
                    query = 'phase:fix_analysis'
                    if query not in seen:
                        seen.add(query)
                        plan.append(query)
                if 'edit' in pc.text.lower() or 'phase 6' in pc.text.lower():
                    query = 'phase:fix_implementation'
                    if query not in seen:
                        seen.add(query)
                        plan.append(query)

        # Keyword queries from claim text
        for claim in claims:
            if claim.claim_type == 'reasoning':
                plan.append('keyword:analysis,root_cause,bug,because')
            if claim.claim_type == 'edit':
                plan.append('keyword:edit,str_replace,patch')

        return plan[:10]  # cap to avoid excessive retrieval


def _obligation_matches(claim: Claim, precondition: Precondition) -> bool:
    """Heuristic: does this precondition apply to this claim?"""
    if precondition.category == 'workflow':
        return claim.claim_type in ('edit', 'workflow')
    if precondition.category == 'reachability':
        return claim.claim_type in ('edit', 'action', 'localization')
    if precondition.category == 'evidence':
        return claim.claim_type in ('reasoning', 'localization', 'edit')
    if precondition.category == 'leakage':
        return claim.claim_type in ('localization', 'reasoning', 'edit')
    return False


# ---------------------------------------------------------------------------
# LLM-assisted claim extractor
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL)
_JSON_BARE_RE = re.compile(r'\{.*\}', re.DOTALL)


def _extract_json_str(text: str) -> str | None:
    """Best-effort JSON string extraction from LLM output."""
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1)
    m = _JSON_BARE_RE.search(text)
    if m:
        return m.group(0)
    return None


class ClaimExtractor:
    """LLM-assisted claim and precondition extractor.

    Falls back to ``ProgrammaticClaimExtractor`` when the LLM call fails or
    the response is not parseable JSON.
    """

    PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'prompts')

    def __init__(self, llm: 'LLM') -> None:  # type: ignore[name-defined]
        self.llm = llm
        self._fallback = ProgrammaticClaimExtractor()
        self.max_json_retries = max(
            int(os.environ.get('VERIFIER_EXTRACTOR_JSON_RETRIES', '2')), 0,
        )

        from jinja2 import Environment, FileSystemLoader

        self._jinja_env = Environment(
            loader=FileSystemLoader(self.PROMPTS_DIR),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def extract(
        self,
        proposal_text: str,
        oracle_preconditions: list[dict] | None = None,
        issue_text: str = '',
        step_index: int = 0,
        history_summary: dict | None = None,
    ) -> ExtractionResult:
        """Extract claims via LLM, falling back to programmatic extractor."""
        prompt = self._render_prompt(
            proposal_text=proposal_text,
            oracle_preconditions=oracle_preconditions,
            step_index=step_index,
            history_summary=history_summary or {},
        )

        for attempt in range(self.max_json_retries + 1):
            try:
                response = self.llm.completion(
                    messages=[{'role': 'user', 'content': prompt}],
                )
                raw_text = response.choices[0].message.content or ''
            except Exception as exc:
                from openhands.core.logger import openhands_logger as logger

                logger.warning(
                    f'[ClaimExtractor] LLM call failed: {exc}. Using fallback.'
                )
                return self._fallback_extract(
                    proposal_text, oracle_preconditions, issue_text, raw_response=str(exc),
                )

            result = self._parse_llm_response(raw_text, proposal_text, issue_text)
            if result is not None:
                result.raw_llm_response = raw_text
                result.raw_llm_prompt = prompt
                # Merge oracle explicit preconditions (LLM only infers)
                result.explicit_preconditions = (
                    self._fallback._build_explicit_preconditions(oracle_preconditions)
                )
                # Build proof obligations from all preconditions
                result.proof_obligations = self._fallback._build_obligations(
                    result.claims, result.all_preconditions,
                )
                return result

        # All retries exhausted
        return self._fallback_extract(
            proposal_text, oracle_preconditions, issue_text,
            raw_response='JSON parse retries exhausted',
        )

    def _render_prompt(
        self,
        proposal_text: str,
        oracle_preconditions: list[dict] | None,
        step_index: int,
        history_summary: dict,
    ) -> str:
        template = self._jinja_env.get_template('extract_claims.j2')
        return template.render(
            proposal_text=proposal_text,
            oracle_preconditions=oracle_preconditions or [],
            step_index=step_index,
            phases_completed=history_summary.get('phases_seen', []),
            files_read=history_summary.get('files_read', []),
            files_edited=history_summary.get('files_edited', []),
            has_think=history_summary.get('has_think', False),
            has_edit=history_summary.get('has_edit', False),
            has_test_after_edit=history_summary.get('has_test_after_edit', False),
        )

    def _parse_llm_response(
        self,
        raw_text: str,
        proposal_text: str,
        issue_text: str,
    ) -> ExtractionResult | None:
        json_str = _extract_json_str(raw_text)
        if json_str is None:
            return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        # Parse claims
        claims: list[Claim] = []
        for i, c in enumerate(data.get('claims', [])):
            claims.append(Claim(
                claim_id=str(c.get('claim_id', f'c{i + 1}')),
                claim_type=str(c.get('claim_type', 'action')),
                text=str(c.get('text', '')),
                file_paths=[str(p) for p in c.get('file_paths', [])],
                symbols=[str(s) for s in c.get('symbols', [])],
            ))

        if not claims:
            return None  # Must have at least one claim

        # Enrich claims with parsed action parameters (deterministic,
        # independent of LLM output — always parse from raw proposal text)
        action_params = _extract_action_params(proposal_text)
        if action_params:
            for claim in claims:
                if claim.claim_type in ('action', 'localization', 'edit'):
                    claim.action_parameters = action_params

        # Parse inferred preconditions
        inferred: list[Precondition] = []
        for i, p in enumerate(data.get('inferred_preconditions', [])):
            inferred.append(Precondition(
                precondition_id=str(p.get('precondition_id', f'p{i + 100}')),
                source='inferred',
                text=str(p.get('text', '')),
                category=str(p.get('category', 'evidence')),
            ))

        # Parse retrieval plan
        retrieval_plan = [str(q) for q in data.get('retrieval_plan', [])][:8]

        return ExtractionResult(
            claims=claims,
            inferred_preconditions=inferred,
            retrieval_plan=retrieval_plan,
        )

    def _fallback_extract(
        self,
        proposal_text: str,
        oracle_preconditions: list[dict] | None,
        issue_text: str,
        raw_response: str = '',
    ) -> ExtractionResult:
        result = self._fallback.extract(
            proposal_text=proposal_text,
            oracle_preconditions=oracle_preconditions,
            issue_text=issue_text,
        )
        result.raw_llm_response = f'[FALLBACK] {raw_response}'
        return result
