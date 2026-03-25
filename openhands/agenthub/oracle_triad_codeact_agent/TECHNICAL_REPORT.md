# Oracle Triad CodeAct Agent: A Hybrid Neuro-Symbolic Architecture for Oracle-Guided SWE-Bench Trajectory Generation

**Module Location:** `openhands/agenthub/oracle_triad_codeact_agent/`  
**Version:** 1.0  

---

## Abstract

We present the **Oracle Triad CodeAct Agent**, a three-component architecture for generating high-fidelity debugging trajectories on the SWE-bench benchmark. The system addresses a fundamental tension in oracle-guided trajectory synthesis: exploiting ground-truth knowledge (patches, test cases) to steer an agent toward the correct fix, while ensuring the resulting trajectory appears independently discoverable — free of oracle-leaked information.

The architecture decomposes the problem into three roles — a *Blinded Debugger*, an *Oracle Planner*, and a *Proposal Validator* — and introduces a **hybrid neuro-symbolic verification pipeline** that combines deterministic symbolic rule evaluation with focused LLM adjudication for ambiguous cases. The verification pipeline operates over a **Structured History Memory** — a bounded, indexed representation of the agent's interaction history — and enforces constraints across five rule families: workflow ordering (A), reachability (B), oracle leakage (C), evidence sufficiency (D), and discoverability (E).

Key contributions include: (1) a triad separation-of-concerns architecture that eliminates direct oracle contamination, (2) a 4.5-stage verification pipeline with deterministic fast-path rejection and selective LLM escalation, (3) a structured claim-evidence framework for auditable trajectory validation, and (4) comprehensive instrumentation for post-hoc analysis.

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [System Architecture](#2-system-architecture)
3. [Blinded Debugger](#3-blinded-debugger)
4. [Oracle Planner](#4-oracle-planner)
5. [Structured History Memory](#5-structured-history-memory)
6. [Claim Extraction (Stage 1)](#6-claim-extraction-stage-1)
7. [Evidence Retrieval (Stage 2)](#7-evidence-retrieval-stage-2)
8. [Symbolic Rule Engine (Stage 3)](#8-symbolic-rule-engine-stage-3)
9. [LLM-Assisted Rule Resolution (Stage 3.5)](#9-llm-assisted-rule-resolution-stage-35)
10. [Verdict Synthesis (Stage 4)](#10-verdict-synthesis-stage-4)
11. [Feedback and Retry Mechanism](#11-feedback-and-retry-mechanism)
12. [React Fact Tracking](#12-react-fact-tracking)
13. [Prompt Engineering](#13-prompt-engineering)
14. [Computational Analysis](#14-computational-analysis)
15. [Graceful Degradation](#15-graceful-degradation)
16. [File Index](#16-file-index)

---

## 1. Introduction and Motivation

### 1.1 Problem Setting

SWE-bench evaluates software engineering agents by requiring them to resolve real GitHub issues in their original repository environments. For supervised fine-tuning (SFT) data generation, one approach is to use an oracle — endowed with the ground-truth patch and test — to guide a debugging agent toward the correct fix. However, naïve oracle injection produces trajectories that contain unrealistic leaps of knowledge: the agent appears to "magically" locate bug-relevant files, methods, and fix patterns without any observable evidence chain.

Such trajectories are detrimental for SFT because they train models to produce outputs that depend on information unavailable at inference time. The resulting models learn shortcuts rather than genuine debugging reasoning.

### 1.2 Desiderata

A trajectory generation system must satisfy three properties simultaneously:

| Property | Description |
|----------|-------------|
| **Correctness** | The trajectory must lead to the ground-truth fix |
| **Non-leakage** | Every claim, localization, and code edit must be traceable to publicly observable evidence |
| **Naturalness** | The debugging workflow must follow a plausible human-like phase ordering |

### 1.3 Approach Overview

We decompose the problem via a **separation-of-concerns** architecture:

- The **Blinded Debugger** generates realistic candidate actions using only the issue description and interaction history — it has *no oracle access*.
- The **Oracle Planner** has full oracle access and selects the best candidate or proposes a revised action, grounding guidance in structured investigation facts with declared preconditions.
- The **Proposal Validator** receives only the interaction history (no oracle access) and verifies that planner proposals are non-leaky and history-grounded before they enter the trajectory.

This triad design ensures that oracle knowledge flows exclusively through the planner, and every planner output is subjected to independent non-leakage verification before materializing as an agent action.

---

## 2. System Architecture

### 2.1 High-Level Control Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    OracleTriadCodeActAgent.step(state)                          │
│                                                                                 │
│  ┌──────────────────────────────────────────┐                                   │
│  │  Phase 1: Candidate Generation           │                                   │
│  │  Blinded Debugger LLM × N candidates     │  ← No oracle access              │
│  └──────────────┬───────────────────────────┘                                   │
│                 ↓                                                                │
│  ┌──────────────────────────────────────────┐                                   │
│  │  Phase 2: Oracle Planning                │                                   │
│  │  Select candidate OR propose revision    │  ← Full oracle context            │
│  │  Grounded in react facts + history       │                                   │
│  └──────────────┬───────────────────────────┘                                   │
│                 ↓                                                                │
│  ┌──────────────────────────────────────────┐     ┌─────────────────────────┐   │
│  │  Phase 3: Proposal Validation            │────→│ 4.5-Stage Verification  │   │
│  │  (only for planner proposals)            │←────│ Pipeline (see §6–§10)   │   │
│  └──────────────┬───────────────────────────┘     └─────────────────────────┘   │
│                 ↓                                                                │
│  ┌──────────────────────────────────────────┐                                   │
│  │  Phase 4: Materialization / Fallback     │                                   │
│  │  Inject guidance → Debugger executes     │                                   │
│  └──────────────────────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Information Barriers

| Component | Issue Text | Interaction History | Oracle Patch | Oracle Test | React Facts | Deep Analysis |
|-----------|:----------:|:-------------------:|:------------:|:-----------:|:-----------:|:-------------:|
| Blinded Debugger | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Oracle Planner | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Proposal Validator | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |

### 2.3 Retry and Fallback Protocol

When the validator rejects a planner proposal:

1. Structured feedback is generated (failed rules, unjustified knowledge, remediation suggestions).
2. Feedback is forwarded to the planner for a revised proposal attempt.
3. Up to `MAX_RETRIES` revision cycles are permitted (default: 2).
4. If all retries are exhausted, the system falls back to the planner's best-rated candidate from Phase 1.
5. If the planner itself fails (LLM error, JSON parse failure), the system falls back to candidate index 0.

This guarantees that the agent always produces an action, preserving liveness.

---

## 3. Blinded Debugger

The Blinded Debugger is the primary CodeAct LLM, receiving the standard SWE-bench prompt (issue description, repository context, tool definitions) and the condensed interaction history. It generates `N` candidate responses per step (configurable via `BLINDED_DEBUGGER_NUM_CANDIDATES`, default: 3).

**Key property:** The debugger has *zero* oracle access. Its candidates represent plausible next actions that a real debugging agent might take. Diversity across candidates increases the probability that at least one is well-aligned with the oracle's steering direction.

**Materialization:** When the planner proposes a revised action (rather than selecting a candidate), the proposal is materialized by injecting it as guidance into the debugger's message history and calling the debugger LLM once more. This ensures the final action passes through the same tool-calling interface as regular CodeAct actions, maintaining format consistency.

---

## 4. Oracle Planner

### 4.1 Design Principles

The Oracle Planner operates with full oracle context — the ground-truth patch, test patch, issue understanding, deep analysis, and structured react facts. Its mandate is to select the best candidate or propose a superior alternative that steers toward the correct fix.

However, the planner must not simply dictate the fix. Its proposals must be *groundable* — every file path, symbol, code pattern, and line reference must either appear in the publicly visible interaction history or be reasonably inferable from it. The planner prompt (§13.1) enforces this through explicit instructions and structured phase ordering constraints.

### 4.2 Decision Protocol

The planner produces a JSON response with the following schema:

```json
{
  "decision": "candidate | proposal",
  "best_candidate_index": "<int>",
  "chosen_candidate_index": "<int | null>",
  "reason": "<short explanation>",
  "proposal_response": "<REASONING + [TOOL CALL] if proposing>",
  "referenced_fact_ids": ["stage1_0", "stage2_1"]
}
```

- **`decision = "candidate"`**: The planner selects an existing candidate (identified by `chosen_candidate_index`). No validation is required since the candidate was generated by the blinded debugger.
- **`decision = "proposal"`**: The planner proposes a new action. This triggers the validation pipeline (§6–§10) before materialization.

### 4.3 React Fact Integration

Each fact in the react fact set has:
- **`fact_id`**: Unique identifier (e.g., `stage1_0`, `stage2_3`)
- **`stage`**: Investigation phase (e.g., "Initial Exploration", "Root Cause Analysis")
- **`goal`**: What the fact aims to achieve
- **`fact`**: The investigative step description
- **`preconditions`**: History states that must hold before this fact is relevant
- **`reasoning`**: Why this step is appropriate at this stage
- **`action`**: The concrete tool call
- **`observation`**: Expected outcome

Facts are presented to the planner grouped by stage. The planner declares which facts it references via `referenced_fact_ids`, enabling downstream tracking and precondition enforcement.

---

## 5. Structured History Memory

### 5.1 Data Model

The verification pipeline operates over a **Structured History Memory** — a bounded, indexed representation of the agent's event history. Each event pair (action + observation) is compressed into a `HistoryUnit`:

```
HistoryUnit
├── unit_id: int              # Sequential index (0, 1, 2, …)
├── unit_type: str            # 'base_instruction' | 'initial_user_instruction' | 'action_observation'
├── action_type: str          # 'CmdRunAction' | 'FileReadAction' | 'FileEditAction' | …
├── action_summary: str       # Short description (≤120 chars)
├── action_text: str          # Full action content (untruncated)
├── observation_text: str     # Paired observation (output, file content, etc.)
├── files_mentioned: list     # Extracted via regex (_FILE_PATH_RE)
├── symbols_mentioned: list   # Extracted via regex (_PYTHON_SYMBOL_RE, 7 capture groups)
├── phase_hint: str           # Inferred phase (reading|running|exploration|…|final_review)
└── tags: set                 # Semantic tags (think, edit, grep, test_run, file_read, …)
```

### 5.2 Indexing

Events are transformed into `HistoryUnit` objects via `StructuredHistoryMemory.from_events()`:

1. **Observation pairing**: Observations are matched to their causal action via `event._cause` links. A fallback look-ahead (up to 3 events) handles missing cause links.
2. **Action extraction**: Each action type maps to a specific text/summary/tag combination (e.g., `FileEditAction` → tag `edit`, summary `EDIT: {path}`).
3. **Observation extraction**: Command output, file content, and edit results are extracted with appropriate tags.
4. **Enrichment**: File paths and Python symbols are extracted via regex patterns from the combined action+observation text.
5. **Phase inference**: A priority-based heuristic assigns a debugging phase to each unit based on tags and content keywords.

### 5.3 Retrieval Interface

The memory supports five search methods — all operating in-memory with O(N) scans over the unit list:

| Method | Input | Scoring |
|--------|-------|---------|
| `keyword_search(keywords, top_k)` | List of keywords | TF-like: `Σ (1 + log(count)) / log(text_len + 1)` |
| `file_path_search(pattern)` | Path substring | Case-insensitive substring match on `files_mentioned` |
| `phase_search(phase)` | Phase name | Exact match on `phase_hint` |
| `tag_search(tags)` | Set of tags | Intersection: `unit.tags ∩ input_tags ≠ ∅` |
| `get_unit(unit_id)` | Integer ID | Direct lookup from `_id_map` |

Convenience accessors aggregate results across the full history:
- `get_all_known_files()` — union of all read, edited, and searched files
- `get_completed_phases()` — ordered list of phases observed so far
- `has_think_action()`, `has_edit_action()`, `has_test_run_after_edit()` — boolean workflow state queries

### 5.4 Symbol Extraction

Python symbols are extracted via a regex with **seven capture groups**, designed to catch common patterns in debugging traces and code output:

| Group | Pattern | Example Match |
|-------|---------|---------------|
| G1 | `class|def <name>` | `class Lookup`, `def resolve_expression` |
| G2 | `isinstance(x, <Type>)` | `isinstance(obj, QuerySet)` |
| G3 | `<name>(` (function calls) | `Lookup.apply()` |
| G4 | `<Name>Error` | `ValueError`, `FieldError` |
| G5 | `: <Type>` (type annotations) | `: Expression` |
| G6 | `= <Module.func>(` | `= models.CharField(` |
| G7 | `<PascalCase.dotted>` | `QuerySet.filter` |

### 5.5 Phase Inference

Each `HistoryUnit` is assigned a debugging phase via heuristic priority ordering:

| Priority | Condition | Phase |
|----------|-----------|-------|
| 1 | `think` tag + analysis keywords (>100 chars) | `fix_analysis` |
| 2 | `edit` tag (not test file) | `fix_implementation` |
| 3 | `test_edit` tag | `test_creation` |
| 4 | `test_run` tag + verification keywords | `verification` |
| 5 | `test_run` tag (no verification keywords) | `running` |
| 6 | `grep` or `search` tag | `exploration` |
| 7 | `file_read` tag | `exploration` |
| 8 | Reproduction script pattern | `test_creation` |
| 9 | `diff` tag | `final_review` |
| 10 | `finish` action | `final_review` |

The canonical phase ordering is: `reading → running → exploration → test_creation → fix_analysis → fix_implementation → verification → final_review`.

---

## 6. Claim Extraction (Stage 1)

### 6.1 Overview

Stage 1 decomposes a planner proposal into a structured set of **claims** and **preconditions**, along with a **retrieval plan** for evidence gathering. Two extractors are available:

- **LLM-based (`ClaimExtractor`):** Renders the `extract_claims.j2` template and parses the LLM's JSON response. Falls back to programmatic extraction on failure.
- **Programmatic (`ProgrammaticClaimExtractor`):** Regex-based extraction of file paths, symbols, tool calls, and claim classification by keyword patterns. Used as fallback or when `VERIFIER_PROGRAMMATIC_ONLY=1`.

### 6.2 Claim Taxonomy

Each claim is classified into one of five types:

| Type | Description | Example |
|------|-------------|---------|
| `action` | Tool invocation (non-edit) | "Run grep for `resolve_expression` in django/db/" |
| `edit` | File modification | "Replace `self.lhs` with `self.lhs.resolve_expression()`" |
| `reasoning` | Bug analysis / root cause | "The bug occurs because Lookup doesn't call resolve_expression" |
| `workflow` | Phase transition | "Verification is complete, proceed to final review" |
| `localization` | File/method targeting | "The relevant code is in `django/db/models/lookups.py`" |

### 6.3 Action Parameter Extraction

For each claim, the extractor deterministically extracts **action parameters** from the proposal text:

- **`view_range: [start, end]`** — Line range from `[TOOL CALL]` JSON arguments
- **`line_numbers: list[int]`** — Sorted, unique line references from regex patterns (e.g., "lines 746–810", "line 123")
- **`search_terms: list[str]`** — Grep/search patterns extracted from bash commands
- **`old_str: str`** — Edit target string from `str_replace` tool arguments (if >10 chars)

### 6.4 Preconditions

Two sources of preconditions:

1. **Explicit preconditions**: Derived from oracle react facts — each fact's `preconditions` field declares what history state must hold.
2. **Inferred preconditions**: Derived automatically from claims:
   - *Reachability*: File paths/symbols not in issue text → must have been discovered in history
   - *Workflow*: Edit claims → Phase 5 analysis must precede Phase 6 editing
   - *Evidence*: Reasoning/localization claims → must have supporting evidence
   - *Leakage*: Localization claims → need observable evidence chain

### 6.5 Retrieval Plan

The extractor generates a bounded retrieval plan (up to 8 queries) using a typed query language:

```
file:<path>         → file_path_search(path)
keyword:<terms>     → keyword_search(terms, top_k=5)
phase:<phase>       → phase_search(phase)
tag:<tag>           → tag_search({tag})
symbol:<name>       → keyword_search([name], top_k=3)
```

---

## 7. Evidence Retrieval (Stage 2)

Stage 2 executes the retrieval plan against the Structured History Memory. The retrieval process:

1. Iterates over the plan queries, dispatching each to the appropriate memory search method.
2. Performs additional **derived retrieval** — for each claim's file paths (up to 3) and symbols (up to 3), performs lookups even if not in the explicit plan.
3. Deduplicates results by `unit_id` via a dictionary accumulator.
4. Returns units sorted by `unit_id` for deterministic ordering.

The retrieval stage is purely in-memory and completes in sub-100ms for typical history sizes (<500 units).

---

## 8. Symbolic Rule Engine (Stage 3)

### 8.1 Design Philosophy

The Symbolic Rule Engine applies deterministic, interpretable rules to the extracted claims against the retrieved evidence. Each rule maps to a precisely defined constraint and produces a binary pass/fail result with a machine-readable reason. This provides:

- **Auditability**: Every rejection traces to a specific rule violation.
- **Speed**: Regex and set operations complete in sub-second time.
- **Consistency**: Identical inputs always produce identical results — no LLM stochasticity.

For rules where symbolic checking is inherently incomplete (e.g., assessing whether a symbol could be "reasonably inferred"), the engine flags the result for **LLM-assisted resolution** (Stage 3.5) rather than making a potentially incorrect deterministic judgment.

### 8.2 RuleResult Data Structure

```python
@dataclass
class RuleResult:
    rule_id: str            # e.g., "B4"
    rule_family: str        # 'workflow' | 'reachability' | 'leakage' | 'evidence' | 'discoverability'
    passed: bool
    severity: str           # 'high' | 'medium' | 'low'
    reason: str
    evidence: list[str]     # Supporting unit_ids or descriptions
    needs_llm_assist: bool  # If True, Stage 3.5 adjudicates before final verdict
    llm_context: dict       # Structured context for resolve_rule.j2 (question, evidence, unjustified items)
```

### 8.3 Rule Families

#### Family A — Workflow Phase Ordering (Severity: High)

These rules enforce that the agent follows a plausible debugging workflow. All are deterministic (no LLM escalation).

| Rule | Constraint | Check |
|------|-----------|-------|
| **A1** | Edit requires analysis | Phase 5 (`fix_analysis`) must precede Phase 6 (`fix_implementation`). Adequate analysis = `think` action with ≥80 chars matching `_ANALYSIS_INDICATORS` regex. |
| **A2** | Verification requires implementation | Phase 7 (`verification`) must have a preceding Phase 6 edit. |
| **A3** | Finalization requires verification | Phase 8 (`final_review`) must have a preceding test run. Detects Django-specific patterns: `runtests.py`, `manage.py test`. |
| **A4** | Phase completion requires evidence | Claims asserting phase completion must have at least one relevant tag in history. Severity: medium. |

#### Family B — Reachability (Severity: High/Medium)

These rules verify that every file path, symbol, edit target, and action parameter in the proposal has a visible provenance in the interaction history or issue text.

| Rule | Constraint | LLM Escalation | Check |
|------|-----------|:---------------:|-------|
| **B1** | File path justification | No | Every file path must appear in: (a) issue text (normalized), (b) `known_files` set, or (c) retrieved units. Exploratory actions (read/view/grep) are exempt. |
| **B2** | Symbol justification | **Yes** | Every symbol must appear in: (a) issue text (case-insensitive, dotted decomposition), (b) `symbols_mentioned` across history, or (c) full text of units. Unjustified symbols → `needs_llm_assist=True`. |
| **B3** | Edit target justification | No | Every file targeted by an edit must have been previously read or searched (`get_all_read_files() ∪ get_all_searched_files()`). |
| **B4** | Action parameter justification | **Yes** | Line numbers must be derivable from context (grep output line numbers, traceback references, cat -n output). `view_range[0] < 20` auto-passes (file header inspection). `view_range[1]` auto-justified if within 150 lines of start. Search terms undergo keyword stripping (removing `\b`, `-w`, quotes, `-e` flags). |

**Path normalization**: `_path_in_text()` implements suffix-based matching to handle workspace prefix mismatches (e.g., `/workspace/django__django__3.1/django/db/` vs. `django/db/`). Paths are decomposed into suffix chains of decreasing length; a suffix match of ≥2 components suffices.

#### Family C — Leakage Detection (Severity: High, All LLM-Escalated)

These rules detect oracle-derived knowledge that lacks public provenance. All three rules escalate to LLM because distinguishing genuine inference from oracle leakage requires semantic judgment.

| Rule | Constraint | Check |
|------|-----------|-------|
| **C1** | Hidden implementation detail | Code snippets in the proposal (extracted from backtick/quoted regions) must appear in the issue text or history. Unsupported snippets → `needs_llm_assist`. |
| **C2** | Unsupported localization | File+method localization claims must have an observable evidence chain. Exploratory localizations (read/view/search) are exempt. |
| **C3** | Oracle-only dependence | Claims with specific paths/symbols must have at least partial public support (file or symbol visible in issue or history). Claims with no support → `needs_llm_assist`. |

#### Family D — Evidence Sufficiency (Severity: Medium)

| Rule | Constraint | LLM Escalation | Check |
|------|-----------|:---------------:|-------|
| **D1** | Bug-cause support | **Yes** | Reasoning claims must reference files/symbols visible in history or issue text. |
| **D2** | Test claim support | No | Test-related claims must have corresponding `test_run` tagged units. |
| **D3** | Analysis claim support | No | Analysis claims must have corresponding `think` tagged units. |

#### Family E — Discoverability (Severity: Low, Advisory Only)

| Rule | Constraint | Check |
|------|-----------|-------|
| **E1** | Discoverable next step | Advisory: recommends inspection before editing. |
| **E2** | Missing prerequisite redirect | Suggests safer discovery steps when prerequisites are unmet. |

### 8.4 Decision Logic After Stage 3

```python
deterministic_high = get_failed_high_excluding_llm_assist(results)
if deterministic_high:
    # Rules A1, A2, A3, B1, B3 failed → instant reject, skip LLM
    → Stage 4 (fast path): verdict = 'invalid'
else:
    # No deterministic rejects → proceed to LLM resolution
    → Stage 3.5: resolve B2, B4, C1, C2, C3, D1
    → Recalculate failed_high and uncertain after resolution
    → Stage 4
```

This two-tier strategy avoids unnecessary LLM calls for clear-cut violations while preserving LLM judgment for genuinely ambiguous cases.

---

## 9. LLM-Assisted Rule Resolution (Stage 3.5)

### 9.1 Motivation

Six rules (B2, B4, C1, C2, C3, D1) involve assessments that symbolic checking cannot reliably make:
- Can a symbol be *reasonably inferred* from visible code patterns?
- Could a line number be *derived* from output the agent has seen?
- Is a code snippet *derivable* from the issue description and error traces?

Rather than accepting all ambiguous cases (high false-negative rate) or rejecting all (high false-positive rate), we escalate each to a focused LLM adjudication call with structured context.

### 9.2 Resolution Protocol

For each rule with `needs_llm_assist=True`:

1. **Render `resolve_rule.j2`** with:
   - The rule's `rule_id`, `severity`, `reason`
   - The rule's `llm_context` (structured: `question`, `unjustified_symbols`, `unjustified_params`, `unsupported_snippets`, etc.)
   - `proposal_text` (excerpt)
   - `issue_text` (excerpt)
   - `evidence_snippets` — list of retrieved units with `{unit_id, action_type, phase_hint, action_summary, text_snippet}`

2. **Make a one-shot LLM call** and parse the response:
   ```json
   {"verdict": "justified | overruled", "reason": "one-sentence explanation"}
   ```
   - `overruled`: The symbolic failure is a false positive; the claim is actually supported. Rule flips to `passed=True`.
   - `justified`: The symbolic failure is genuine. Rule remains `passed=False`.

3. **Fail-open on error**: If the LLM call fails (network, parsing), the rule defaults to `passed=True`. This is a conservative design choice that prevents infrastructure failures from blocking trajectory generation.

### 9.3 Post-Resolution

After all LLM-assist rules are resolved:
- `needs_llm_assist` is cleared on all results.
- `failed_high` is recalculated using the standard `get_failed_high()` method.
- `uncertain` (C-family failures that survived LLM resolution) proceeds to Stage 4 for final synthesis.

---

## 10. Verdict Synthesis (Stage 4)

### 10.1 Deterministic Fast Paths

| Condition | Verdict | Rationale |
|-----------|---------|-----------|
| Any `failed_high` rules | `invalid` | High-severity violation with no mitigation |
| All rules passed, no uncertain | `valid` | No objections from any rule family |

### 10.2 LLM-Assisted Synthesis

When `uncertain` rules exist (C-family partial failures that survived symbolic + LLM resolution), the `synthesize_verdict.j2` template is rendered with:

- All rule results (passed and failed)
- Extracted claims
- Evidence summaries (capped at 20 units)
- Issue and proposal text

The LLM returns:
```json
{
  "verdict": "valid | invalid | uncertain",
  "reason": "one-paragraph justification",
  "suspected_leakage": ["item1", "item2"],
  "suggestion": "concrete next step (empty if valid)"
}
```

### 10.3 VerificationVerdict Structure

The final output encapsulates the complete audit trail:

```python
@dataclass
class VerificationVerdict:
    step_index: int
    verdict: str                          # 'valid' | 'invalid' | 'uncertain'
    reason: str
    claims: list[dict]                    # Extracted claims
    explicit_preconditions: list[dict]
    inferred_preconditions: list[dict]
    retrieval_queries: list[str]
    retrieved_unit_ids: list[int]
    rule_results: list[dict]              # All 14+ rules with pass/fail and reasons
    failed_obligations: list[str]
    suspected_leakage: list[str]
    feedback_message: str                 # For planner retry
    suggestion: str                       # Remediation guidance
    raw_extraction_response: str          # Stage 1 LLM raw output
    raw_synthesis_response: str           # Stage 4 LLM raw output
```

---

## 11. Feedback and Retry Mechanism

When a proposal is rejected, the verifier constructs a structured feedback message:

```
[QA REVIEW - ORACLE PROPOSAL REJECTED]

Reason: <synthesis reason>

Failed rule checks:
  - B4 (high): Line numbers [746, 810] not inferrable from visible context
  - C2 (high): File+method localization of lookups.py:resolve_expression lacks evidence chain

Suspected leakage:
  - Specific knowledge of resolve_expression method signature

Suggestion: Read the file around lines where Lookup class is defined before referencing specific methods.

Revise the proposal with these hard constraints:
  1. Do not mention or imply oracle/golden patch content.
  2. Keep all claims grounded in observed history only.
  3. Use strictly incremental next steps.
```

This feedback is forwarded to the planner via the planner prompt's `planner_feedback` field, enabling the planner to adjust its proposal while respecting the identified constraints.

---

## 12. React Fact Tracking

### 12.1 Overview

`ReactFactTracker` manages a structured set of investigation facts derived from preprocessing. Each fact represents one step in a plausible investigation trajectory, complete with preconditions, reasoning, and expected observations.

### 12.2 Usage Protocol

1. **Initialization**: Facts are loaded from the oracle context JSON (`react_facts` key) and assigned unique IDs (`{stage}_{index}`).
2. **Presentation**: Available (unused) facts are rendered to the planner prompt grouped by investigation stage.
3. **Consumption**: When the planner declares `referenced_fact_ids`, those facts are marked as used and excluded from future prompts.
4. **Precondition Forwarding**: Referenced facts' preconditions are passed to the validator, enriching the explicit precondition set.
5. **Summary**: At the end of each step, a usage summary is logged:
   ```json
   {"total_facts": 12, "used_facts": 4, "remaining_facts": 8, "used_fact_ids": ["stage1_0", "stage2_1", ...]}
   ```

### 12.3 Fact Structure

```json
{
  "fact_id": "stage2_1",
  "stage": "Root Cause Analysis",
  "goal": "Identify why Lookup doesn't resolve expressions",
  "fact": "Open django/db/models/lookups.py and read the Lookup class",
  "preconditions": ["File django/db/models/lookups.py must be visible in history"],
  "reasoning": "Lookup class inherits Expression but may not call resolve_expression",
  "action": "[TOOL CALL] read_file({\"path\": \"django/db/models/lookups.py\", \"view_range\": [1, 50]})",
  "observation": "Lookup class definition with __init__ and as_sql methods"
}
```

---

## 13. Prompt Engineering

### 13.1 Prompt Templates

Five Jinja2 templates orchestrate the LLM interactions:

| Template | Stage | Purpose | LLM Calls |
|----------|-------|---------|-----------|
| `planner_select_or_propose.j2` | Planner decision | Present candidates, oracle context, react facts; request selection or proposal | 1 per step |
| `extract_claims.j2` | Stage 1 | Decompose proposal into typed claims, preconditions, and retrieval plan | 1 (with fallback) |
| `resolve_rule.j2` | Stage 3.5 | Adjudicate ambiguous symbolic rule failure | 0–6 per proposal |
| `synthesize_verdict.j2` | Stage 4 | Synthesize final verdict from rule results and evidence | 0–1 per proposal |
| `validate_oracle_proposal.j2` | Legacy critic | One-shot validation (used when `PROPOSAL_VALIDATOR=critic`) | 1 per proposal |

### 13.2 Planner Prompt Structure

The planner prompt follows a 6-section layout designed for information locality:

1. **§1 Role & Task** — System-level instructions, non-leakage constraints
2. **§2 Issue** — The GitHub issue description
3. **§3 Oracle Context** — Private context (patch, analysis) + react facts (grouped by stage)
4. **§4 Interaction History** — Compact session summary with file/command inventories
5. **§5 Candidates** — Indexed list of blinded debugger responses
6. **§6 Decision Rules & Output Format** — Phase ordering constraints, SFT quality rules, JSON schema

### 13.3 Template Safety

All templates use `dict`-style access for `llm_context` fields (e.g., `rule.llm_context['key']`, `'key' in rule.llm_context`) to avoid Jinja2 attribute resolution errors on plain dict objects.

---

## 14. Computational Analysis

### 14.1 Per-Proposal Cost

| Stage | Operations | Latency (typical) |
|-------|-----------|-------------------|
| Stage 1 (Extraction) | 1 LLM call | ~10–15s |
| Stage 2 (Retrieval) | O(Q × N) in-memory scans | <100ms |
| Stage 3 (Symbolic) | O(C × R) regex/set operations | <500ms |
| Stage 3.5 (LLM Resolution) | 0–6 LLM calls | 0–90s |
| Stage 4 (Synthesis) | 0–1 LLM call | 0–15s |

Where Q = number of retrieval queries (~8), N = history units, C = number of claims, R = number of rules (14+).

### 14.2 Best and Worst Case

- **Best case** (deterministic reject): Stages 1–3 only → 1 LLM call, <16s
- **Best case** (all pass, no uncertain): Stages 1–3 → 1 LLM call, <16s
- **Worst case** (all 6 LLM-assist rules + uncertain synthesis): 1 + 6 + 1 = 8 LLM calls, ~120s

In practice, most proposals trigger 1–3 LLM-assist rules, yielding 2–5 total LLM calls per validation.

### 14.3 Memory Complexity

The Structured History Memory holds O(N) units where N is the number of action-observation pairs in the trajectory. Each unit stores the full text of the action and observation (no truncation), ensuring retrieval accuracy at the cost of linear memory in trajectory length.

---

## 15. Graceful Degradation

| Failure | Fallback Behavior |
|---------|-------------------|
| Planner LLM error | Return `best_candidate_index=0` |
| Planner JSON parse failure | Retry up to `max_json_parse_retries` (default: 3), then `best_candidate_index=0` |
| Verifier initialization failure | Fall back to legacy critic; if critic also fails, candidate-only mode |
| Claim extraction LLM failure | Fall back to `ProgrammaticClaimExtractor` |
| LLM-resolution call failure | Fail-open: treat rule as overruled (`passed=True`) |
| Verdict synthesis LLM failure | Deterministic fallback based on rule results |
| All validation retries exhausted | Use planner's best candidate with warning |
| No candidates generated | Use empty response (should not occur) |

---

## 16. File Index

| File | Lines | Role |
|------|------:|------|
| `oracle_triad_codeact_agent.py` | ~920 | Main triad orchestration agent |
| `oracle_planner.py` | ~370 | Oracle planner + `ReactFactTracker` + `PlannerDecision` |
| `proposal_critic.py` | ~260 | Legacy one-shot blinded critic |
| `verifier.py` | ~940 | 4.5-stage verification pipeline |
| `symbolic_rules.py` | ~1100 | Deterministic rule engine (families A–E) |
| `history_memory.py` | ~550 | Structured history indexing and retrieval |
| `claim_extractor.py` | ~650 | Claim, precondition, and retrieval plan extraction |
| `prompts/planner_select_or_propose.j2` | ~200 | Planner decision prompt |
| `prompts/extract_claims.j2` | ~140 | Claim extraction prompt |
| `prompts/resolve_rule.j2` | ~110 | LLM rule adjudication prompt |
| `prompts/synthesize_verdict.j2` | ~140 | Verdict synthesis prompt |
| `prompts/validate_oracle_proposal.j2` | ~160 | Legacy critic prompt |

### Evaluation Entry Points

| File | Role |
|------|------|
| `evaluation/benchmarks/swe_bench_optimized/run_infer_oracle_triad.py` | Python evaluation runner |
| `evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh` | Shell launcher with env var management |

---

## Appendix A: Complete Rule Reference

| ID | Family | Severity | LLM Assist | One-line Description |
|----|--------|----------|:----------:|----------------------|
| A1 | Workflow | High | No | Edit requires prior analysis (≥80 char think action with analysis indicators) |
| A2 | Workflow | High | No | Verification requires prior implementation (edit) |
| A3 | Workflow | High | No | Finalization requires prior verification (test run; detects runtests.py, manage.py test) |
| A4 | Workflow | Medium | No | Phase completion assertions require evidence |
| B1 | Reachability | High | No | File paths must be in issue text, known files, or retrieved units (exploratory exempt) |
| B2 | Reachability | Medium | Yes | Symbols must be visible in issue or history (unjustified → LLM judgment) |
| B3 | Reachability | High | No | Edit targets must be previously read or searched |
| B4 | Reachability | High | Yes | Line numbers derivable from context; search terms keyword-stripped; exploratory reads exempt |
| C1 | Leakage | High | Yes | Code snippets must appear in issue or history (unsupported → LLM judgment) |
| C2 | Leakage | High | Yes | File+method localization needs observable chain (exploratory exempt) |
| C3 | Leakage | High | Yes | Claims need partial public support (paths/symbols in issue or history) |
| D1 | Evidence | Medium | Yes | Reasoning claims need file/symbol evidence in history or issue |
| D2 | Evidence | Medium | No | Test claims need test_run tagged units |
| D3 | Evidence | Medium | No | Analysis claims need think tagged units |
| E1 | Discoverability | Low | No | Advisory: inspect before editing |
| E2 | Discoverability | Low | No | Advisory: suggest safer discovery steps |

## Appendix B: Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `BLINDED_DEBUGGER_NUM_CANDIDATES` | 3 | Number of debugger candidates per step |
| `ORACLE_PLANNER_MAX_RETRIES` | 2 | Planner proposal revision retries on validation failure |
| `ORACLE_PLANNER_LLM_CONFIG` | `oracle_planner` | LLM config section name for planner |
| `ORACLE_PLANNER_JSON_PARSE_MAX_RETRIES` | 3 | JSON parsing retries for planner response |
| `ORACLE_PROPOSAL_CRITIC_LLM_CONFIG` | `blinded_critic` | LLM config section name for critic/verifier |
| `ORACLE_PROPOSAL_CRITIC_JSON_PARSE_MAX_RETRIES` | 3 | JSON parsing retries for critic response |
| `PROPOSAL_VALIDATOR` | `verifier` | Validation backend: `verifier` \| `critic` \| `none` |
| `USE_LEGACY_CRITIC` | `0` | Legacy flag: `1` → `PROPOSAL_VALIDATOR=critic` |
| `ORACLE_PLANNER_CONTEXT_PATH` | — | Path to oracle context JSON (set per instance) |
| `ORACLE_PREPROCESS_DIR` | Auto-detect | Directory containing `{id}_react_facts.json` and `{id}_analysis.md` |
| `VERIFIER_PROGRAMMATIC_ONLY` | `0` | Force programmatic claim extractor (skip LLM) |
| `VERIFIER_EXTRACTOR_JSON_RETRIES` | 2 | LLM JSON parsing retries for claim extraction |
| `VERIFIER_LLM_CONFIG` | Uses critic config | LLM config for verifier |
| `ORACLE_PLANNER_SAVE_PROMPTS` | `0` | Enable planner prompt saving |
| `ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS` | `0` | Enable critic/verifier prompt saving |
| `ORACLE_PLANNER_SAVE_PROMPTS_DIR` | — | Output directory for planner prompts |
| `ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS_DIR` | — | Output directory for critic/verifier prompts |

## Appendix C: Key Regex Patterns

```
_ANALYSIS_INDICATORS:
  'the (bug|issue|problem) (is|occurs|happens)' | 'root cause' | 'because' |
  'the fix (is|should|would)' | 'needs? to be? (changed|fixed|replaced|updated)' |
  'should (return|use|be|handle)'

_EXPLORATORY_ACTION_RE:
  'view|read|examine|inspect|look at|open|cat|head|tail|less|
   grep|find|search|locate|list|ls|tree|check|see|explore|discover|
   navigate|investigate|trace|browse'

_PYTHON_SYMBOL_RE (7 groups):
  G1: class|def <name>              G5: : <ClassName>
  G2: isinstance(x, <Type>)        G6: = <Module.func>(
  G3: <name>(                       G7: <PascalCase.dotted>
  G4: <Name>Error

_FILE_PATH_RE:     (/[\w._-]+){2,}(\.\w+)?
_TEST_FILE_RE:     test_\w+|conftest|\w+_test\.py$
_RUNTESTS_CMD_RE:  .*runtests\.py
_MANAGE_TEST_CMD:  manage\.py test
```
