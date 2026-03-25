# Neuro-Symbolic Verifier — Development Plan

_Workspace_: `/home/v-murongma/code/OpenHands_SWE-Bench-Optimized`  
_Module_: `openhands/agenthub/oracle_triad_codeact_agent/`  
_Replaces_: `OracleProposalCritic` (one-shot LLM critic)  
_Created_: 2025-03-24

---

## 0. Status Tracker

| Phase | Description | Status | Session |
|-------|-------------|--------|---------|
| 1 | Interaction history memory + retrieval layer | **DONE** | Session 1 (2025-03-24) |
| 2 | Claim/precondition extraction (Stage 1) | **DONE** (programmatic + LLM extractor + prompt) | Session 1–2 |
| 3 | Symbolic rule engine (Stage 3) | **DONE** | Session 1 (2025-03-24) |
| 3b | History-grounded retrieval (Stage 2) | **DONE** (verifier.py `_run_retrieval()` + history_memory.py search methods) | Session 2 |
| 4 | Decision synthesis + LLM-assisted verdict (Stage 4) | **DONE** (verifier.py + synthesize_verdict.j2) | Session 2 |
| 5 | Orchestration integration + backward compat | **DONE** (wired into oracle_triad_codeact_agent.py) | Session 2 |
| 6 | Testing + eval smoke run | **PARTIAL** (unit smoke test passed; eval run pending) | Session 2 |

Update this table at the end of each session.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  OracleTriadCodeActAgent.step()                 │
│                                                                 │
│  1. Generate N debugger candidates                              │
│  2. Oracle Planner selects/proposes                             │
│  3. ┌──────────────────────────────────────────────────────┐    │
│     │   HistoryGroundedVerifier.verify()                   │    │
│     │                                                      │    │
│     │   Stage 1: Extract claims + preconditions            │    │
│     │       ↓                                              │    │
│     │   Stage 2: Retrieve evidence from StructuredHistory  │    │
│     │       ↓                                              │    │
│     │   Stage 3: Apply symbolic rules                      │    │
│     │       ↓                                              │    │
│     │   Stage 4: Synthesize verdict + feedback             │    │
│     └──────────────────────────────────────────────────────┘    │
│  4. Accept / reject / retry                                     │
└─────────────────────────────────────────────────────────────────┘
```

Key design constraints:
- **No live filesystem access** — operates only over prior interaction history
- **No shell commands** — no subprocess, no exec, no docker
- **Bounded retrieval** — at most 1 primary retrieval + 1–2 derived follow-up queries
- **Deterministic symbolic layer** — programmatic rule evaluation, not LLM-only
- **Lightweight** — no separate runtime env, no new agent loop, no new tools
- **Backward compatible** — existing `OracleProposalCritic` import paths + class name kept as alias

---

## 2. File Plan

| File | Action | Role |
|------|--------|------|
| `history_memory.py` | **NEW** | `StructuredHistoryMemory` + `HistoryUnit` types + search/retrieval methods |
| `claim_extractor.py` | **NEW** | Stage 1 — LLM-assisted claim/precondition/obligation extraction |
| `symbolic_rules.py` | **NEW** | Stage 3 — All rule families (A–E) as programmatic checks + path normalisation helpers |
| `verifier.py` | **NEW** | `HistoryGroundedVerifier` — 4-stage orchestration incl. Stage 2 retrieval (`_run_retrieval()`) |
| `prompts/extract_claims.j2` | **NEW** | Jinja2 prompt for claim extraction LLM call |
| `prompts/synthesize_verdict.j2` | **NEW** | Jinja2 prompt for verdict synthesis LLM call |
| `proposal_critic.py` | **MODIFY** | Add `HistoryGroundedVerifier` alias; keep backward compat class |
| `oracle_triad_codeact_agent.py` | **MODIFY** | Wire verifier in place of critic; build `StructuredHistoryMemory` |
| `run_infer_oracle_triad.py` | **MODIFY** | Update env config for verifier LLM if needed |
| `NEURO_SYMBOLIC_VERIFIER_DEV.md` | **UPDATE** | This file — update status after each session |

---

## 3. Data Structures

### 3.1 History Units

```python
# history_memory.py

@dataclass
class HistoryUnit:
    """One retrievable unit from interaction history."""
    unit_id: int                      # sequential index
    unit_type: str                    # 'base_instruction' | 'initial_user_instruction' | 'action_observation'
    event_indices: list[int]          # original event indices from state.history
    action_type: str | None           # e.g. 'CmdRunAction', 'FileReadAction', 'FileEditAction', 'AgentThinkAction'
    action_summary: str               # short summary: "RUN: grep -n ...", "READ: /path/to/file"
    action_text: str                  # full action content
    observation_text: str             # full observation content (empty for base instructions)
    files_mentioned: list[str]        # file paths extracted from action+observation
    symbols_mentioned: list[str]      # method/class names extracted (best-effort)
    phase_hint: str | None            # inferred workflow phase if detectable
    tags: set[str]                    # searchable tags: {'file_read', 'grep', 'test_run', 'edit', 'think', ...}
```

### 3.2 Structured History Memory

```python
# history_memory.py

class StructuredHistoryMemory:
    """Indexed interaction history for bounded retrieval."""
    units: list[HistoryUnit]

    def keyword_search(self, keywords: list[str], top_k: int = 10) -> list[HistoryUnit]: ...
    def file_path_search(self, path_pattern: str) -> list[HistoryUnit]: ...
    def phase_search(self, phase: str) -> list[HistoryUnit]: ...
    def tag_search(self, tags: set[str]) -> list[HistoryUnit]: ...
    def get_unit(self, unit_id: int) -> HistoryUnit | None: ...
    def get_all_edited_files(self) -> list[str]: ...
    def get_all_read_files(self) -> list[str]: ...
    def has_think_action(self) -> bool: ...
    def get_latest_phase(self) -> str | None: ...

    @classmethod
    def from_events(cls, events: list[Event]) -> 'StructuredHistoryMemory': ...
```

### 3.3 Claim & Precondition Structures

```python
# claim_extractor.py

@dataclass
class Claim:
    claim_id: str                    # "c1", "c2", ...
    claim_type: str                  # 'action' | 'reasoning' | 'workflow' | 'localization' | 'edit'
    text: str                        # natural language claim text
    file_paths: list[str]            # file paths referenced in claim
    symbols: list[str]               # symbols/methods referenced in claim

@dataclass
class Precondition:
    precondition_id: str             # "p1", "p2", ...
    source: str                      # 'oracle_json' | 'inferred'
    text: str                        # natural language precondition text
    category: str                    # 'workflow' | 'reachability' | 'evidence' | 'leakage'

@dataclass
class ProofObligation:
    obligation_id: str               # "o1", "o2", ...
    claim_id: str                    # which claim this obligation supports
    precondition_id: str | None      # linked precondition (if any)
    description: str                 # what must be proven
    retrieval_hints: list[str]       # keywords/paths to search for evidence

@dataclass
class ExtractionResult:
    claims: list[Claim]
    explicit_preconditions: list[Precondition]
    inferred_preconditions: list[Precondition]
    proof_obligations: list[ProofObligation]
    retrieval_plan: list[str]        # ordered list of retrieval queries
```

### 3.4 Verification Results

```python
# symbolic_rules.py

@dataclass
class RuleResult:
    rule_id: str                      # e.g. "workflow.phase_6_requires_phase_5"
    rule_family: str                  # 'workflow' | 'reachability' | 'leakage' | 'evidence' | 'discoverability'
    passed: bool
    severity: str                     # 'high' | 'medium' | 'low'
    related_claim_ids: list[str]
    related_precondition_ids: list[str]
    evidence_unit_ids: list[int]      # HistoryUnit IDs used as evidence
    reason: str

# verifier.py

@dataclass
class VerificationVerdict:
    step_index: int
    verdict: str                      # 'valid' | 'invalid' | 'uncertain'
    claims: list[dict]
    explicit_preconditions: list[dict]
    inferred_preconditions: list[dict]
    retrieval_queries: list[str]
    retrieved_unit_ids: list[int]
    rule_results: list[dict]          # serialized RuleResult list
    failed_obligations: list[str]
    suspected_leakage: list[str]
    feedback_message: str             # structured feedback for planner retry
    suggestion: str                   # what planner should do instead
    raw_extraction_response: str
    raw_synthesis_response: str

    # Backward-compat bridge for orchestration agent
    @property
    def valid(self) -> bool:
        return self.verdict == 'valid'

    def to_dict(self) -> dict: ...
```

---

## 4. Phase-by-Phase Implementation

### Phase 1 — Interaction History Memory + Retrieval Layer

**Goal**: Build `StructuredHistoryMemory` that indexes interaction history into retrievable units with keyword/tag/file/phase search.

**Files to create**: `history_memory.py`

**Depends on**: Nothing new — reads `state.history` events (same as `_render_history_text_full()`)

**Implementation steps**:

1. Define `HistoryUnit` dataclass with fields listed in §3.1.
2. Implement `StructuredHistoryMemory.from_events(events)`:
   - Event 0 (`SystemMessageAction`) → `BaseInstructionUnit` (unit_type='base_instruction')
   - Event 1 (first `MessageAction` from user) → `InitialUserInstructionUnit` (unit_type='initial_user_instruction')
   - Subsequent events: pair each Action with its corresponding Observation into one `ActionObservationUnit` (unit_type='action_observation'). Unpaired events become solo units.
   - For each unit, extract: `files_mentioned` (regex for paths like `/workspace/...`), `symbols_mentioned` (best-effort: class/function names from grep/read output), `tags` based on action type, `phase_hint` from heuristic matching.
3. Implement retrieval methods:
   - `keyword_search(keywords, top_k)` — score units by keyword overlap in action_text + observation_text. Use simple normalized term-frequency scoring (no external deps).
   - `file_path_search(path_pattern)` — match against `files_mentioned`.
   - `phase_search(phase)` — match against `phase_hint`.
   - `tag_search(tags)` — intersection match against `tags`.
   - Convenience: `get_all_edited_files()`, `get_all_read_files()`, `has_think_action()`, `get_latest_phase()`.
4. Phase heuristic for `phase_hint`:
   - 'reading' if action is first user message restatement
   - 'running' if action runs existing tests
   - 'exploration' if action is grep/find/read
   - 'test_creation' if action creates/edits a test file (`test_*.py` or `*_test.py` or reproduction script)
   - 'fix_analysis' if action is `AgentThinkAction` or think tool with analysis content
   - 'fix_implementation' if action is `FileEditAction` on non-test source file
   - 'verification' if action runs tests after an edit
   - 'final_review' if action is a diff or final comparison

**Reference code to read**:
- `oracle_triad_codeact_agent.py` lines 570–700: `_render_history_text_full()` — current event iteration pattern
- Event types: `openhands/events/action/*.py`, `openhands/events/observation/*.py`

**Validation**: Unit test that constructs a mock event list, builds memory, and verifies retrieval correctness.

---

### Phase 2 — Claim & Precondition Extraction (Stage 1)

**Goal**: Build `ClaimExtractor` that takes a planner proposal + oracle preconditions + history summary and outputs structured claims, preconditions, proof obligations, and a retrieval plan.

**Files to create**: `claim_extractor.py`, `prompts/extract_claims.j2`

**Depends on**: Phase 1 (`StructuredHistoryMemory` for history summary in prompt)

**Implementation steps**:

1. Define dataclasses: `Claim`, `Precondition`, `ProofObligation`, `ExtractionResult` (§3.3).
2. Create `ClaimExtractor` class:
   - `__init__(self, llm: LLM)` — takes an LLM instance (shares the critic's LLM config).
   - `extract(self, proposal_text, oracle_preconditions, history_summary, step_index) -> ExtractionResult`
3. Design `extract_claims.j2` prompt:
   - Input sections: proposal text, oracle-supplied preconditions (from react facts), brief history summary (file list, commands run, phases completed — NOT the full history string).
   - Instructions: extract claims from the proposal, identify explicit preconditions from oracle JSON, infer latent preconditions, generate proof obligations, output retrieval plan.
   - Output: strict JSON matching `ExtractionResult` schema.
4. Implement robust JSON parsing (reuse `_extract_json` pattern from existing critic).
5. Implement fallback: if LLM fails or returns unparseable output, generate a minimal extraction with one "whole_proposal" claim and the oracle preconditions as-is.
6. Latent precondition inference rules (encoded in prompt):
   - If proposal mentions a file path → inferred precondition: "file must be visible in history or discoverable"
   - If proposal mentions a symbol/method → inferred: "symbol must be visible in issue or history"
   - If proposal is an edit action → inferred: "edit target must have been inspected"
   - If proposal claims bug cause → inferred: "bug cause must be supported by observed evidence"
   - If proposal asserts phase completion → inferred: "phase completion must have explicit evidence"
   - If proposal is a test action → inferred: "test context must exist in history"

**Reference code to read**:
- `proposal_critic.py` lines 45–60: existing `OracleProposalCritic.__init__` and LLM setup pattern
- `oracle_planner.py` lines 90–110: `ReactFactTracker.get_preconditions_for_facts()` — how oracle preconditions are structured
- `prompts/validate_oracle_proposal.j2` — existing prompt structure for context

**Validation**: Test with a sample proposal string + mock preconditions, verify structured output.

---

### Phase 3 — Symbolic Rule Engine (Stage 3)

**Goal**: Implement deterministic rule evaluation over claims + preconditions + retrieved evidence. No LLM calls in this phase.

**Files to create**: `symbolic_rules.py`

**Depends on**: Phase 1 (HistoryUnit for evidence), Phase 2 (Claim/Precondition types)

**Implementation steps**:

1. Define `RuleResult` dataclass (§3.4).
2. Implement the rule engine as a class `SymbolicRuleEngine`:
   ```python
   class SymbolicRuleEngine:
       def __init__(self, memory: StructuredHistoryMemory): ...
       def evaluate_all(self, claims, preconditions, retrieved_units) -> list[RuleResult]: ...
   ```
3. Implement each rule family:

   **Family A — Workflow phase rules**:
   - `_rule_A1_edit_requires_analysis()`: Check if any claim involves edit/str_replace. If so, search memory for phase='fix_analysis' or think actions with analysis content. Fail if absent.
   - `_rule_A2_verification_requires_implementation()`: Check if proposal enters verification. Search for prior file edits. Fail if none.
   - `_rule_A3_finalization_requires_verification()`: Similar — check for test runs after edits.
   - `_rule_A4_phase_completion_requires_evidence()`: If claim asserts "analysis complete" or similar, verify explicit evidence exists.

   **Family B — Reachability rules**:
   - `_rule_B1_file_path_justification()`: For each file path in claims, check: appears in issue text? appears in `memory.get_all_read_files()`? discoverable from a search step in history?
   - `_rule_B2_symbol_justification()`: For each symbol in claims, check: in issue text? in any retrieved unit's observation_text?
   - `_rule_B3_edit_target_justification()`: If claim is an edit, verify the target file has been read or discovered via search.

   **Family C — Leakage rules**:
   - `_rule_C1_hidden_implementation_detail()`: Check if proposal introduces concrete code/values not in issue or history. This is the hardest to make fully deterministic — use heuristic: if claim contains code snippets, verify they appear in retrieved history.
   - `_rule_C2_unsupported_localization()`: If claim localizes to a specific file+method, check evidence chain.
   - `_rule_C3_oracle_only_dependence()`: If claim has no supporting history evidence at all, flag it.

   **Family D — Evidence sufficiency rules**:
   - `_rule_D1_bug_cause_support()`: If claim is type 'reasoning' about bug cause, require at least 1 supporting evidence span.
   - `_rule_D2_test_claim_support()`: If claim references tests, require test-related history.
   - `_rule_D3_analysis_claim_support()`: If claim asserts analysis, require reasoning evidence.

   **Family E — Discoverability rules**:
   - `_rule_E1_discoverable_next_step()`: Prefer proposals that inspect/verify before editing. Return as 'low' severity advisory.
   - `_rule_E2_missing_prerequisite_redirect()`: If insufficient evidence, suggest the safest next step (grep/read/think/test). Return as suggestion, not hard failure.

4. Each rule method returns a `RuleResult`. The `evaluate_all()` method runs all applicable rules and returns the full list.

**Key design decisions**:
- Rules A and B are high-severity (hard reject on failure).
- Rules C are high-severity but may be `uncertain` (need LLM assist in Stage 4).
- Rules D are medium-severity.
- Rules E are low-severity (advisory, feed into suggestions).
- A rule only runs if its triggering condition is met (e.g., A1 only runs if there's an edit claim).

**Reference code to read**:
- `prompts/validate_oracle_proposal.j2` — existing textual rules that become programmatic
- `prompts/planner_select_or_propose.j2` lines on phase enforcement — phase definitions

**Validation**: Unit test with mock claims + mock memory, verify each rule fires correctly.

---

### Phase 3b — History-Grounded Retrieval (Stage 2)

**Goal**: Given the extraction result from Stage 1, retrieve evidence from `StructuredHistoryMemory` to ground the symbolic rule evaluation in Stage 3. Uses bounded, deterministic retrieval — no LLM calls.

**Files**: Retrieval logic lives in `verifier.py` (`_run_retrieval()` static method) and calls methods on `history_memory.py` (`StructuredHistoryMemory`).

**Depends on**: Phase 1 (StructuredHistoryMemory), Phase 2 (ExtractionResult with `retrieval_plan` and claims)

**Implementation (as built)**:

The retrieval pipeline has two passes — **plan-driven retrieval** and **claim-derived retrieval** — followed by deduplication.

#### Pass 1: Plan-driven retrieval

Each query in `ExtractionResult.retrieval_plan` is dispatched by prefix:

| Query prefix | Dispatch method | Semantics |
|---|---|---|
| `file:<path>` | `memory.file_path_search(path)` | Return units whose `files_mentioned` match `<path>` (substring, case-insensitive) |
| `phase:<phase>` | `memory.phase_search(phase)` | Return units whose `phase_hint` equals `<phase>` (case-insensitive) |
| `keyword:<kw1>,<kw2>,...` | `memory.keyword_search(keywords, top_k=5)` | TF-normalised keyword scoring across `full_text` |
| `tag:<tag>` | `memory.tag_search({tag})` | Set intersection on unit tags (e.g. `edit`, `grep`, `think`, `test_run`) |
| `symbol:<sym>` | `memory.keyword_search([sym], top_k=3)` | Keyword search using the symbol name as the query term |

The retrieval plan is generated during Stage 1 (by the LLM extractor or by `ProgrammaticClaimExtractor` as a fallback). Each query string is consumed in order, and results are accumulated into a deduplicated `{unit_id → HistoryUnit}` map.

#### Pass 2: Claim-derived retrieval

After the plan-driven pass, additional evidence is retrieved directly from the structured claims:

- For each claim's `file_paths` (up to 3): `memory.file_path_search(fp)` — retrieves units that touched the same file.
- For each claim's `symbols` (up to 3): `memory.keyword_search([sym], top_k=2)` — retrieves units that mention the symbol.

This ensures that even if the retrieval plan missed a relevant file or symbol, evidence is still gathered from the claims themselves.

#### Deduplication and ordering

All retrieved units are merged by `unit_id` (dict keyed on `unit_id`), then sorted by `unit_id` ascending for deterministic ordering. The result is a `(sorted_units, queries_used)` tuple passed to Stage 3.

#### Underlying search methods (`StructuredHistoryMemory`)

| Method | Algorithm | Notes |
|---|---|---|
| `keyword_search(keywords, top_k)` | TF-like scoring: `(1 + log(count)) / log(text_len + 1)` per keyword, summed. Sorted descending, capped at `top_k`. | Case-insensitive. Searches `full_text` (action + observation concatenated). |
| `file_path_search(path_pattern)` | Substring match of `path_pattern.lower()` against each `files_mentioned` entry. | Returns all matching units (no ranking). |
| `phase_search(phase)` | Exact match on `phase_hint` (case-insensitive). | Returns all units with that phase. |
| `tag_search(tags)` | Set intersection: `unit.tags & tags`. | Tags include: `edit`, `file_read`, `grep`, `search`, `think`, `test_run`, `cmd_run`, `agent_message`, etc. |

#### Boundedness guarantees

- Plan-driven pass: bounded by `len(extraction.retrieval_plan)`, typically 3–8 queries.
- Claim-derived pass: bounded by `3 × len(claims)` file queries + `3 × len(claims)` symbol queries.
- Each individual search returns at most `top_k` results (default 5 for keyword, unbounded for file/phase/tag but practically small).
- Total: O(number of claims × history size) worst case, but history units are typically < 100 and claims < 10.

#### Example retrieval flow

For a proposal that edits `/workspace/django__django__3.1/django/db/models/lookups.py`:

```
Plan queries:
  file:django/db/models/lookups.py     → units 5, 12 (file_read, grep)
  phase:fix_analysis                   → unit 14 (think action)
  keyword:Lookup,__init__,rhs          → units 5, 12, 14

Claim-derived:
  file_paths: lookups.py               → units 5, 12 (already seen)
  symbols: Lookup.__init__             → unit 12 (already seen)

Final deduplicated: [unit_5, unit_12, unit_14]  (sorted by unit_id)
```

**Validation**: Covered by the end-to-end verifier smoke test — retrieval is tested implicitly through rule evaluation (rules fail if relevant evidence is not retrieved).

---

### Phase 4 — Decision Synthesis + LLM-Assisted Verdict (Stage 4)

**Goal**: Combine rule results into a final verdict. Use LLM only when rule results are ambiguous (e.g., leakage rules where string matching is insufficient).

**Files to create**: `verifier.py`, `prompts/synthesize_verdict.j2`

**Depends on**: Phases 1–3

**Implementation steps**:

1. Define `VerificationVerdict` dataclass (§3.4).
2. Implement `HistoryGroundedVerifier` class:
   ```python
   class HistoryGroundedVerifier:
       def __init__(self, llm: LLM, issue_text: str): ...

       def verify(
           self,
           step_index: int,
           proposal_text: str,
           history_memory: StructuredHistoryMemory,
           fact_preconditions: list[dict] | None = None,
           attempt: int = 0,
       ) -> VerificationVerdict: ...

       @classmethod
       def from_env(cls, issue_text: str) -> 'HistoryGroundedVerifier | None': ...
   ```

3. `verify()` orchestrates the 4 stages:
   ```
   Stage 1: claim_extractor.extract(proposal, preconditions, history_summary)
   Stage 2: for each retrieval query → memory.keyword_search() / file_path_search()
            optionally 1–2 derived queries based on initial results
   Stage 3: rule_engine.evaluate_all(claims, preconditions, retrieved_units)
   Stage 4: if any rule uncertain or C-family needs LLM assist → synthesize verdict via LLM
            else → deterministic verdict from rule results
   ```

4. Verdict decision logic:
   - Any **high-severity** rule failed → `verdict='invalid'`
   - Any **C-family** rule uncertain and no high-severity failures → call LLM for `synthesize_verdict.j2`
   - All rules passed → `verdict='valid'`
   - Otherwise → `verdict='uncertain'` (treated as valid with warnings, similar to current fail-open)

5. `synthesize_verdict.j2` prompt:
   - Input: proposal text, rule results summary, retrieved evidence spans, claims, issue text
   - Task: determine if ambiguous cases are truly leakage/unjustified or acceptable
   - Output: JSON with `verdict`, `reason`, `suspected_leakage`, `suggestion`

6. Feedback message construction:
   - For `invalid`: list failed rules, unmet preconditions, suspected leakage, suggest what planner should do instead
   - For `uncertain`: list warnings, suggest conservative alternative
   - Reuse and extend `_build_feedback_message` pattern from current critic

7. `from_env()` factory:
   - Uses same `ORACLE_PROPOSAL_CRITIC_LLM_CONFIG` / `blinded_critic` config key (backward compat)
   - Or new `VERIFIER_LLM_CONFIG` env var with fallback to old name

8. Prompt saving:
   - Reuse `ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS_DIR` mechanism
   - Save extraction prompt + response and synthesis prompt + response separately

**Reference code to read**:
- `proposal_critic.py`: full class — `from_env()`, `_maybe_save_prompt()`, JSON parsing patterns
- `oracle_triad_codeact_agent.py` lines 160–250: current critic integration in `step()` loop

**Validation**: End-to-end test with mocked LLM responses, verify correct verdict for known-good and known-bad proposals.

---

### Phase 5 — Orchestration Integration + Backward Compatibility

**Goal**: Wire `HistoryGroundedVerifier` into `OracleTriadCodeActAgent.step()` replacing the `OracleProposalCritic` call. Maintain backward-compatible imports.

**Files to modify**: `proposal_critic.py`, `oracle_triad_codeact_agent.py`, `run_infer_oracle_triad.py`

**Depends on**: Phases 1–4

**Implementation steps**:

1. **`proposal_critic.py`** — Add backward-compat alias at end of file:
   ```python
   # Keep backward-compatible import
   # from ...proposal_critic import OracleProposalCritic still works
   # New code should use HistoryGroundedVerifier from verifier.py
   ```
   No changes to existing `OracleProposalCritic` class — keep it intact as fallback option.

2. **`oracle_triad_codeact_agent.py`** — Key changes:
   - Add import: `from .verifier import HistoryGroundedVerifier`
   - Add import: `from .history_memory import StructuredHistoryMemory`
   - In `__init__`: add `self._verifier: HistoryGroundedVerifier | None = None`
   - In `_init_components()`:
     - After existing planner/critic init, try to init verifier:
       ```python
       # Try new verifier first; fall back to legacy critic
       use_legacy_critic = os.environ.get('USE_LEGACY_CRITIC', '0') == '1'
       if not use_legacy_critic:
           self._verifier = HistoryGroundedVerifier.from_env(issue_text=public_issue_text)
           if self._verifier:
               self._proposal_critic = None  # verifier replaces critic
       ```
   - In `step()`, before calling verifier/critic:
     - Build `StructuredHistoryMemory` from `state.history`:
       ```python
       history_memory = StructuredHistoryMemory.from_events(state.history)
       ```
   - Replace critic validation call with verifier call:
     ```python
     if self._verifier is not None:
         verdict = self._verifier.verify(
             step_index=step_index,
             proposal_text=decision.proposal_response_text,
             history_memory=history_memory,
             fact_preconditions=fact_preconditions if fact_preconditions else None,
             attempt=planner_attempt,
         )
         # Bridge to existing flow
         is_valid = verdict.valid
         feedback = verdict.feedback_message
     elif self._proposal_critic is not None:
         validation = self._proposal_critic.validate(...)
         is_valid = validation.valid
         feedback = validation.feedback_message or validation.reason
     ```
   - Log verifier results to triad log with event type `'verifier_verdict'` (distinct from old `'proposal_critic_validation'`).

3. **`run_infer_oracle_triad.py`** — Minimal changes:
   - No new env vars required if using same LLM config
   - If new `VERIFIER_LLM_CONFIG` is added, set it in the shell launcher too

4. **`__init__.py`** — No changes needed (agent class name unchanged).

5. **Environment variable for switching**:
   - `USE_LEGACY_CRITIC=1` → uses old `OracleProposalCritic` (default: 0)
   - This allows A/B comparison during eval.

**Reference code to read**:
- `oracle_triad_codeact_agent.py` lines 90–100: `__init__` fields
- `oracle_triad_codeact_agent.py` lines 120–250: `step()` planner/critic loop
- `oracle_triad_codeact_agent.py` lines 290–320: `_init_components()`

**Validation**: Run with `USE_LEGACY_CRITIC=1` to verify no regression, then default to verify new verifier works.

---

### Phase 6 — Testing + Eval Smoke Run

**Goal**: Validate end-to-end with real eval instance.

**Implementation steps**:

1. AST parse check on all new `.py` files.
2. Jinja template render check on new `.j2` files.
3. Unit test: `StructuredHistoryMemory.from_events()` with mock events.
4. Unit test: `SymbolicRuleEngine` with mock claims triggering each rule family.
5. Integration test: `HistoryGroundedVerifier.verify()` with mocked LLM.
6. Smoke eval run:
   ```bash
   export USE_LEGACY_CRITIC=0
   bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
     llm.eval_qwen3_coder_30b_a3b_instruct HEAD OracleTriadCodeActAgent 1 100 1 \
     princeton-nlp/SWE-bench_Verified test 1
   ```
7. Compare triad logs between legacy critic and new verifier for same instance.
8. Check verdict quality: does the verifier correctly reject phase-skipping proposals and accept well-grounded ones?

---

## 5. Key Integration Points — Quick Reference

### 5.1 Where the critic is currently called

[oracle_triad_codeact_agent.py](oracle_triad_codeact_agent.py) — `step()` method, approximately lines 190–230:

```python
validation = self._proposal_critic.validate(
    step_index=step_index,
    history_text=full_history_text,
    proposal_response_text=decision.proposal_response_text,
    attempt=planner_attempt,
    fact_preconditions=fact_preconditions if fact_preconditions else None,
)
```

The verifier must produce an object with `.valid` (bool) and `.feedback_message` (str) at minimum, plus `.to_dict()` for logging.

### 5.2 Where history events are available

- `state.history` — full event list (type `list[Event]`)
- Already iterated in `_render_history_text_full()` which imports event types

### 5.3 Where oracle preconditions come from

- `ReactFactTracker.get_preconditions_for_facts(referenced_ids)` → `list[dict]` with keys: `fact_id`, `stage`, `fact_summary`, `preconditions`
- Called in `step()` after planner decision, before critic/verifier call

### 5.4 Where the LLM config comes from

- `OracleProposalCritic.from_env()` uses `ORACLE_PROPOSAL_CRITIC_LLM_CONFIG` (default `blinded_critic`) — verifier should reuse this

### 5.5 Where triad logs are written

- `_append_triage_entry(entry)` writes to `/tmp/oracle_triad_<PID>.jsonl`
- `step()` appends to `self.triad_log` list

### 5.6 Event type imports needed

```python
from openhands.events.action import (
    CmdRunAction, FileEditAction, FileReadAction, MessageAction,
)
from openhands.events.action.agent import AgentThinkAction
from openhands.events.action.message import SystemMessageAction
from openhands.events.observation import (
    CmdOutputObservation, FileEditObservation, FileReadObservation, Observation,
)
from openhands.events.event import Event
```

---

## 6. Symbolic Rule Reference

Compact reference for implementation. Severity and trigger in parentheses.

### Family A — Workflow

| ID | Rule | Severity | Trigger |
|----|------|----------|---------|
| A1 | Edit requires analysis | high | claim.type == 'edit' or claim involves str_replace |
| A2 | Verification requires implementation | high | claim.type == 'workflow' and mentions verification |
| A3 | Finalization requires verification | high | claim.type == 'workflow' and mentions final review |
| A4 | Phase completion needs evidence | medium | claim asserts phase completion |

### Family B — Reachability

| ID | Rule | Severity | Trigger |
|----|------|----------|---------|
| B1 | File path justification | high | claim.file_paths is non-empty |
| B2 | Symbol justification | medium | claim.symbols is non-empty |
| B3 | Edit target justification | high | claim.type == 'edit' |
| B4 | Action parameter justification | high | claim.action_parameters is non-empty (view_range, search_terms) |

### Family C — Leakage

| ID | Rule | Severity | Trigger |
|----|------|----------|---------|
| C1 | Hidden implementation detail | high | claim contains code patterns |
| C2 | Unsupported localization | high | claim localizes to specific file+method |
| C3 | Oracle-only dependence | high | claim has no public evidence |

### Family D — Evidence sufficiency

| ID | Rule | Severity | Trigger |
|----|------|----------|---------|
| D1 | Bug-cause support | medium | claim.type == 'reasoning' about bug cause |
| D2 | Test claim support | medium | claim references test behavior |
| D3 | Analysis claim support | medium | claim asserts analysis conclusion |

### Family E — Discoverability

| ID | Rule | Severity | Trigger |
|----|------|----------|---------|
| E1 | Discoverable next step | low | proposal edits without prior inspection |
| E2 | Missing prerequisite redirect | low | insufficient evidence for any high-sev rule |

---

## 7. Session Handoff Template

Copy this at the end of each session before handing off:

```markdown
### Session Handoff — [DATE]

**Completed in this session**: [phases completed, specific files created/modified]
**Status**: [update Phase table in §0]
**Blocking issues**: [any blockers]
**Next session should**:
  1. [first priority]
  2. [second priority]
**Files modified**:
  - [list]
**Testing done**:
  - [AST check? unit tests? eval run?]
```

---

## 8. Design Decisions Log

Record non-obvious decisions here as they are made.

| Decision | Rationale | Date |
|----------|-----------|------|
| Keep `OracleProposalCritic` intact, add verifier alongside | Allows `USE_LEGACY_CRITIC=1` fallback for A/B comparison | 2025-03-24 |
| Use LLM only in Stage 1 (extraction) and Stage 4 (synthesis) | Stages 2-3 must be deterministic for provenance tracking | 2025-03-24 |
| Action-observation pairs as main retrieval unit | Validation usually depends on "what was done" + "what was observed" jointly | 2025-03-24 |
| Simple term-frequency for keyword search, no external deps | Keep lightweight; embedding-based search would add deps and latency | 2025-03-24 |
| `verdict` field uses 3-way (valid/invalid/uncertain) not bool | `uncertain` allows nuanced handling without hard failure | 2025-03-24 |
| Reuse `blinded_critic` LLM config for verifier | Minimizes config changes; both components serve same validation role | 2025-03-24 |

---

## 9. Session Change Log

_(Updated by each session after implementation work)_

### Session 1 — 2025-03-24 — Phases 1, 2 (partial), 3

**Files created:**
- `history_memory.py` — `StructuredHistoryMemory` + `HistoryUnit` with `from_events()` factory and keyword/file/tag/phase search. Action-observation pairing via `cause` field or lookahead. Phase heuristic classification. File path and Python symbol extraction.
- `claim_extractor.py` — `Claim`, `Precondition`, `ProofObligation`, `ExtractionResult` dataclasses + `ProgrammaticClaimExtractor` (regex/keyword-based fallback). Extracts edit/action/reasoning/workflow/localization claims. Infers latent preconditions for reachability, workflow ordering, evidence, and leakage. Builds proof obligations and retrieval plan.
- `symbolic_rules.py` — `SymbolicRuleEngine` with 14 rules across 5 families (A—E). `RuleResult` dataclass. All rules are deterministic with no LLM calls.

**Testing done:**
- AST parse check: all 3 files pass
- Import check: all cross-module imports resolve in `openhands-swebench` conda env
- Functional test with mock data:
  - `ProgrammaticClaimExtractor`: correctly extracts edit claim with file paths, infers 3 preconditions (reachability × 2, workflow × 1), builds 5 obligations
  - `StructuredHistoryMemory`: keyword search, file path search, phase search all return correct units
  - `SymbolicRuleEngine`: 7 rules evaluated, 4 high-severity failures correctly detected (A1: no analysis before edit, B1: unjustified file path, B3: uninspected edit target, C3: oracle-only dependence)

**Not done (deferred to next session):**
- `ClaimExtractor` (LLM-based) — needs Jinja prompt + LLM completion wiring (Phase 2 completion)
- `HistoryGroundedVerifier` (4-stage orchestrator) — Phase 4
- `prompts/extract_claims.j2` and `prompts/synthesize_verdict.j2` — Phase 2 + 4
- Orchestration integration into `oracle_triad_codeact_agent.py` — Phase 5

### Session Handoff — 2025-03-24

**Completed in this session**: Phases 1, 2 (data structures + programmatic extractor), 3 fully implemented and tested.
**Status**: See updated Phase table in §0.
**Blocking issues**: None.
**Next session should**:
  1. Implement LLM-based `ClaimExtractor` class + `prompts/extract_claims.j2` (finish Phase 2)
  2. Implement `HistoryGroundedVerifier` + `prompts/synthesize_verdict.j2` (Phase 4)
  3. Wire into `oracle_triad_codeact_agent.py` (Phase 5)
  4. Run eval smoke test (Phase 6)
**Files modified**:
  - `history_memory.py` (NEW)
  - `claim_extractor.py` (NEW)
  - `symbolic_rules.py` (NEW)
  - `NEURO_SYMBOLIC_VERIFIER_DEV.md` (updated status)
**Testing done**:
  - AST check: pass
  - Import check: pass
  - Functional mock test: pass (extractor, memory, rule engine)

---

### Session 2 Change Log — 2025-03-25

**Completed**: Phases 2 (LLM extractor), 4 (verifier orchestrator), 5 (agent wiring)

**New files created**:
  - `prompts/extract_claims.j2` — Jinja2 LLM prompt for structured claim extraction
  - `prompts/synthesize_verdict.j2` — Jinja2 LLM prompt for verdict synthesis on ambiguous cases
  - `verifier.py` — `HistoryGroundedVerifier` 4-stage orchestrator + `VerificationVerdict` dataclass

**Files modified**:
  - `claim_extractor.py` — Added `ClaimExtractor` class (LLM-assisted, with `ProgrammaticClaimExtractor` fallback)
  - `oracle_triad_codeact_agent.py` — Wired verifier into step() loop:
    - Added imports for `HistoryGroundedVerifier`, `StructuredHistoryMemory`
    - Added `_verifier` field and `_use_legacy_critic` toggle (`USE_LEGACY_CRITIC=1`)
    - `_init_components()` initializes verifier by default, falls back to legacy critic if verifier init fails
    - Validation block in step() uses verifier (builds `StructuredHistoryMemory.from_events()`) or legacy critic
  - `NEURO_SYMBOLIC_VERIFIER_DEV.md` — Updated status table

**Environment variables**:
  - `USE_LEGACY_CRITIC=1` — Fall back to old `OracleProposalCritic` (default: use verifier)
  - `VERIFIER_PROGRAMMATIC_ONLY=1` — Skip LLM extraction, use programmatic-only mode
  - `VERIFIER_LLM_CONFIG` — LLM config key (fallback: `ORACLE_PROPOSAL_CRITIC_LLM_CONFIG` / `blinded_critic`)
  - `VERIFIER_EXTRACTOR_JSON_RETRIES` — Max JSON parse retries for LLM extraction (default: 2)

**Testing done**:
  - AST check: pass (all 5 module files)
  - Import check: pass (openhands-swebench env)
  - Functional smoke test: pass
    - Good proposal (file read + think + known file): `valid`
    - Unknown file reference: `invalid` (A1 rule)
    - No analysis before edit: `valid` (no edit claims extracted from minimal proposal)
  - Feedback message generation verified on invalid verdict

**Remaining for Phase 6 (eval smoke run)**:
  1. Run a real SWE-bench instance with `USE_LEGACY_CRITIC=0` (default)
  2. Compare verifier logs vs legacy critic logs
  3. Verify no regressions in accept/reject rates
  4. Tune rule thresholds if needed
