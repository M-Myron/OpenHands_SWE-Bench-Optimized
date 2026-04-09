# Oracle Triad Experiment - Module Report

_Workspace_: `/home/v-murongma/code/OpenHands_SWE-Bench-Optimized`  
_Purpose_: triad orchestration with **Blinded Debugger** + **Oracle Planner** + **Blinded Proposal Critic** for SWE-bench trajectories.

---

## 1. Overview

This module extends the prior guided idea into a three-component control loop:

1. **Blinded Debugger** (main `CodeAct` LLM) receives the normal SWE issue prompt and generates multiple candidate responses each step.
2. **Oracle Planner** (separate LLM with private oracle context) inspects the full interaction history + all candidates and either:
   - selects one candidate, or
   - proposes a better next response.
3. **Blinded Proposal Critic** (separate LLM, no oracle access) validates only planner-proposed responses for reachability/non-leakage.

If critic rejects a proposal, feedback is sent back to planner for retry. When retries are exhausted, the system falls back to planner's best candidate.

---

## 2. File Inventory

| File | Role |
|------|------|
| `openhands/agenthub/oracle_triad_codeact_agent/__init__.py` | Registers `OracleTriadCodeActAgent` in `Agent` registry |
| `openhands/agenthub/oracle_triad_codeact_agent/oracle_triad_codeact_agent.py` | Main triad orchestration agent + per-process triad logging helpers |
| `openhands/agenthub/oracle_triad_codeact_agent/oracle_planner.py` | Oracle planner LLM wrapper + `PlannerDecision` parsing + `ReactFactTracker` (graph DAG + legacy) |
| `openhands/agenthub/oracle_triad_codeact_agent/proposal_critic.py` | Blinded proposal critic wrapper + `ProposalValidationResult` |
| `openhands/agenthub/oracle_triad_codeact_agent/prompts/planner_select_or_propose.j2` | Planner prompt template (includes investigation graph facts section) |
| `openhands/agenthub/oracle_triad_codeact_agent/prompts/validate_oracle_proposal.j2` | Proposal critic prompt template (includes fact preconditions section) |
| `evaluation/benchmarks/swe_bench_optimized/run_infer_oracle_triad.py` | Eval entrypoint + per-instance oracle context writer + react facts loader |
| `evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh` | Shell launcher |
| `evaluation/evaluation_outputs/outputs/.../preprocess/{id}/stage3_bridged.json` | Per-instance bridged investigation DAG (v5 format) |
| `evaluation/evaluation_outputs/outputs/.../preprocess/{id}/stage2_facts.json` | Per-instance graph-based investigation DAG (v3 format) |
| `evaluation/evaluation_outputs/outputs/.../preprocess/{id}_react_facts.json` | Per-instance structured react facts (legacy format) |

---

## 3. Core Components

### 3.1 `OracleTriadCodeActAgent`

Class: `OracleTriadCodeActAgent(CodeActAgent)` (`VERSION = '1.0'`)

Key runtime fields:

- `_oracle_planner`: lazy-initialized from env/config
- `_proposal_critic`: lazy-initialized from env/config
- `_num_candidates`: `BLINDED_DEBUGGER_NUM_CANDIDATES` (default `3`)
- `_planner_max_retries`: `ORACLE_PLANNER_MAX_RETRIES` (default `2`)
- `triad_log`: in-memory list mirrored to `/tmp/oracle_triad_<PID>.jsonl`

#### Step flow (`step(state)`)

1. Handle pending actions and `/exit` early exits.
2. Build condensed messages for the debugger LLM.
3. Build **full** history text from `state.history` for planner/critic.
4. Generate `N` debugger candidates by calling primary LLM `N` times.
5. Planner loop (`0.._planner_max_retries`):
   - planner returns decision (`candidate` or `proposal`) plus `best_candidate_index` and `referenced_fact_ids`.
   - extract `referenced_fact_ids` and get preconditions from `ReactFactTracker`.
   - if `candidate`: choose selected/best candidate, mark referenced facts as used, and continue.
   - if `proposal`:
     - if critic disabled, materialize proposal directly and mark facts used;
     - else validate proposal with blinded critic (passing `fact_preconditions`).
     - if critic passes: materialize proposal and mark facts used.
     - if critic fails and retries remain: send feedback back to planner.
     - if retries exhausted: fallback to planner `best_candidate_index`.
6. Log `react_fact_usage_summary` event (total/used/remaining facts).
7. Convert chosen response to actions and return first queued action.

#### Proposal materialization detail

Planner proposal text is injected as a user guidance message via `_inject_planner_guidance()`:

- marker: `[ORACLE PLANNER GUIDANCE - APPROVED BY BLINDED CRITIC]`
- message explicitly describes the two-part structure (REASONING + TOOL CALL) and instructs the debugger to incorporate reasoning and execute the tool call
- then primary debugger LLM is called once with full history + tools to produce executable tool/action output.
- **Post-processing**: if the debugger response has tool_calls but empty content, the REASONING part is extracted from the original proposal and injected as `msg.content` so `response_to_actions()` will use it as the action's thought text.

This keeps tool/action generation in the same interface as `CodeActAgent`.

#### Tool descriptions helper

`_build_tool_descriptions()` iterates `self.tools` (list of `ChatCompletionToolParam`) and produces a human-readable summary:
- Tool name as `### name`
- Description (truncated to 500 chars)
- Parameters with type, required marker, and description (truncated to 200 chars each)

This string is passed to `OraclePlanner` and rendered in the planner prompt's "Available Tools" section.

#### Full-history rendering helper

`_render_history_text_full(events)` includes:

- session index: base instruction events (system message + user message), files read/edited, and commands run
- full event log for actions and observations
- skips `SystemMessageAction` and initial user `MessageAction` in the event body to avoid duplication with SESSION INDEX
- base instruction events are NOT truncated (full content rendered in SESSION INDEX)
- `_truncate_text()` helper remains available for other uses

This is intended to avoid evidence truncation at planner/critic stage (subject to model token limits).

---

### 3.2 `OraclePlanner` (`oracle_planner.py`)

Dataclass: `PlannerDecision`

Fields:

- `step_index: int`
- `decision: 'candidate' | 'proposal'`
- `best_candidate_index: int` (always required)
- `chosen_candidate_index: int | None`
- `reason: str`
- `proposal_response_text: str`
- `raw_planner_response: str`
- `referenced_fact_ids: list[str]` (default `[]`) — IDs of react facts referenced by the planner in this decision

Class: `ReactFactTracker`

Manages structured investigation facts with per-step usage state and DAG-based availability. Supports three data formats:

1. **Bridged graph** (swegym_v5 `stage3_bridged.json`): nodes with categories (`fact`, `bridge_fact`, `organizational_fact`, `plan_fact`, `edit_step`, `validation_step`), single evidence dict per node `{action, observation}`, plus `motivation`, `is_root`, `grounding`, and `discovery_type` fields.
2. **Stage-2 graph** (swegym_v3 `stage2_facts.json`): nodes with categories (`trigger`, `base_fact`, ...), evidence as array of `{reasoning, action, observation}` dicts. Fully backward compatible.
3. **Legacy stage-based** (`_react_facts.json`): flat `stages[].facts[]` structure, internally converted to graph-compatible nodes.

Key API:

- `__init__(react_facts_data: dict | None)` — auto-detects format via `graph` vs `stages` key; normalizes v5 single-dict evidence to 1-element list; parses nodes with IDs like `f1`, `b2`, `e1` (graph) or `phase_3_exploration_0` (legacy)
- `has_facts -> bool`
- `is_graph_mode -> bool` — True if loaded from any graph-based format
- `get_available_nodes() -> list[dict]` — returns nodes that are (a) not fully_used AND (b) all precondition nodes are fully_used. Enforces DAG ordering.
- `mark_facts_used(fact_ids: list[str])` — accepts node IDs (e.g. `"f1"`, `"b2"`). Also supports legacy compound IDs (`"t1:0"`).
- `get_preconditions_for_facts(fact_ids) -> list[dict]` — returns resolved preconditions; in graph mode, precondition node IDs are resolved to include category + statement text
- `render_available_facts_text() -> str` — renders only available (precondition-satisfied, not-fully-used) nodes grouped by category; includes `motivation` field for v5 nodes; fully-used nodes omitted entirely
- `get_usage_summary() -> dict` — returns `{total_nodes, fully_used_nodes, available_nodes, blocked_nodes, total_evidence, used_evidence, ...}`

Node usage states:
- **not_used**: no evidence items consumed
- **partially_used**: some (not all) evidence items consumed — shown with used items marked
- **fully_used**: all evidence items consumed — omitted from planner input; unlocks downstream nodes

Behavior:

- Prompt rendered via `planner_select_or_propose.j2`.
- Expects strict JSON object response.
- Robust parsing fallback:
  - non-JSON / malformed JSON -> fallback `candidate 0`
  - invalid decision -> coerced to `candidate`
  - empty proposal with `decision=proposal` -> fallback to `best_candidate`
- Optional prompt dump:
  - set `ORACLE_PLANNER_SAVE_PROMPTS_DIR`

Factory:

- `from_env(issue_text, oracle_context, tool_descriptions='', react_fact_tracker=None)`
- config key: `ORACLE_PLANNER_LLM_CONFIG` (default `oracle_planner`)
- disables completion logging (`llm_config.log_completions = False`)
- `tool_descriptions` is passed to `_render_prompt` and rendered in the Jinja template "Available Tools" section
- `react_fact_tracker` (if provided with facts) logs fact count at init and passes available facts text to each prompt render

---

### 3.3 `OracleProposalCritic` (`proposal_critic.py`)

Dataclass: `ProposalValidationResult`

Fields:

- `step_index`, `valid`, `reason`
- `unjustified_knowledge`, `prerequisite_conditions`
- `feedback_message`
- `proposal_response_text`
- `raw_critic_response`

Behavior:

- Prompt rendered via `validate_oracle_proposal.j2`.
- Expects strict JSON object response.
- Fail-open behavior on LLM call or parse failure (`valid=True`) to avoid deadlock.
- Enforced invariant:
  - non-empty `unjustified_knowledge` => forced `valid=False`
- Builds rejection feedback for planner retries.
- `validate()` accepts optional `fact_preconditions: list[dict] | None` — when provided, these are rendered in the critic prompt as additional validation criteria for the proposal.
- Optional prompt dump:
  - set `ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS_DIR`

Factory:

- `from_env(issue_text)`
- config key: `ORACLE_PROPOSAL_CRITIC_LLM_CONFIG` (default `blinded_critic`)
- disables completion logging (`llm_config.log_completions = False`)

---

## 4. Prompt Contracts

### 4.1 Planner prompt

File: `planner_select_or_propose.j2`

Input sections:

- public issue text
- private oracle context
- **available tool descriptions** (conditional, from `tool_descriptions` variable)
- **proposal format requirements** (REASONING + TOOL CALL structure)
- **available reference facts** (conditional, from `available_facts_text` when `has_react_facts` is true) — unused facts grouped by stage with IDs, preconditions, recommended reasoning/action
- full interaction history
- debugger candidates
- optional critic feedback

Output JSON schema (required):

```json
{
  "decision": "candidate" | "proposal",
  "best_candidate_index": 0,
  "chosen_candidate_index": 0,
  "reason": "...",
  "proposal_response": "...",
  "referenced_fact_ids": ["f1", "f2", "b1"]
}
```

Important hard rules in prompt:

- planner must **always** output `best_candidate_index`, even when proposing.
- `proposal_response` MUST contain both REASONING text and a `[TOOL CALL]` suggestion using exact tool names from the Available Tools section.
- `referenced_fact_ids` MUST list all consumed node IDs (e.g. `"f1"`, `"b2"`). When proposing, should reference at least one fact if applicable unused facts remain.
- **Fact usage rules**: check preconditions before using a fact, adapt reasoning/action (don't copy verbatim), use facts aggressively when preconditions are met.
- **DAG-gated availability**: only nodes whose preconditions are all fully_used are shown in the prompt. Downstream nodes progressively unlock as upstream nodes are consumed.

### 4.2 Proposal critic prompt

File: `validate_oracle_proposal.j2`

Checks:

- consistency with history
- no logic jumps
- no leakage/oracle mention
- no implementation details absent from history/issue
- **workflow phase ordering**: rejects proposals that skip required phases (e.g., Phase 6 edit without Phase 5 analysis in history).
- **SFT reasoning quality**: rejects proposals whose REASONING text doesn't explain "why"
- **fact precondition validation** (conditional): when `fact_preconditions` list is non-empty, the critic checks whether preconditions of referenced facts are satisfied by the interaction history. Unsatisfied preconditions indicate an unjustified knowledge jump.

Output JSON schema:

```json
{
  "valid": true,
  "reason": "...",
  "unjustified_knowledge": [],
  "prerequisite_conditions": []
}
```

---

## 5. Evaluation Wiring

### 5.1 `run_infer_oracle_triad.py`

Key integration points:

- Imports `openhands.agenthub.oracle_triad_codeact_agent` to register agent.
- Patches:
  - `AGENT_CLS_TO_FAKE_USER_RESPONSE_FN['OracleTriadCodeActAgent'] = ...['CodeActAgent']`
- Default agent class if unspecified:
  - `OracleTriadCodeActAgent`

Per-instance wrapper: `process_instance_oracle_triad`

1. `clear_triage_log()`
2. Build oracle context file (JSON) under:
   - `{eval_output_dir}/oracle_planner_context/{instance_id}.json`
3. Set `ORACLE_PLANNER_CONTEXT_PATH` env var for this instance.
4. Optionally set per-instance prompt dump dirs:
   - planner: `oracle_planner_prompts/{instance_id}`
   - proposal critic: `oracle_proposal_critic_prompts/{instance_id}`
5. Run base `process_instance(...)`.
6. Read and clear triad log from `/tmp/oracle_triad_<PID>.jsonl`.
7. Write per-instance triad log:
   - `{eval_output_dir}/oracle_triad_logs/{instance_id}.jsonl`
8. Attach to `output.test_result['oracle_triad_log']`.

### 5.2 Oracle context payload

Written JSON keys:

- `instance_id`
- `patch`
- `test_patch`
- `issue_understanding`
- `deep_analysis` — markdown from `{instance_id}_analysis.md` (if exists)
- `react_facts` — structured JSON from `{instance_id}_react_facts.json` (if exists)

`issue_understanding` is assembled from optional dataset columns if present:

- `issue_understanding`
- `bug_description`
- `bug_trigger`
- `fix_rationale`
- `hints_text`

Fallback text is used when those columns do not exist.

---

## 6. Shell Launcher

File: `scripts/run_oracle_triad_infer.sh`

Usage:

```bash
bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
  <MODEL_CONFIG> <COMMIT_HASH> [AGENT] [EVAL_LIMIT] [MAX_ITER] \
  [NUM_WORKERS] [DATASET] [SPLIT] [N_RUNS]
```

Defaults:

- `AGENT=OracleTriadCodeActAgent`
- `MAX_ITER=100`
- `NUM_WORKERS=1`
- `DATASET=princeton-nlp/SWE-bench_Lite`
- `SPLIT=test`
- `N_RUNS=1`

Triad env defaults:

- `BLINDED_DEBUGGER_NUM_CANDIDATES=3`
- `ORACLE_PLANNER_MAX_RETRIES=2`
- `ORACLE_PLANNER_LLM_CONFIG=oracle_planner`
- `ORACLE_PROPOSAL_CRITIC_LLM_CONFIG=blinded_critic`
- `ORACLE_PLANNER_SAVE_PROMPTS=0`
- `ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS=0`

---

## 7. Triad Log Schema

The per-instance `oracle_triad_logs/{instance_id}.jsonl` contains mixed event records.

Typical event types:

- `debugger_candidate`
- `oracle_planner_decision`
- `proposal_critic_validation`
- `react_fact_usage_summary`

Representative examples:

```json
{
  "step_index": 3,
  "event": "debugger_candidate",
  "candidate_index": 1,
  "response_text": "..."
}
```

```json
{
  "step_index": 3,
  "event": "oracle_planner_decision",
  "attempt": 0,
  "decision": "proposal",
  "best_candidate_index": 2,
  "chosen_candidate_index": null,
  "reason": "...",
  "proposal_response_text": "...",
  "raw_planner_response": "...",
  "referenced_fact_ids": ["phase_3_exploration_0", "phase_3_exploration_1"]
}
```

```json
{
  "step_index": 3,
  "event": "proposal_critic_validation",
  "attempt": 0,
  "valid": false,
  "reason": "...",
  "unjustified_knowledge": ["..."],
  "prerequisite_conditions": ["..."],
  "feedback_message": "...",
  "proposal_response_text": "...",
  "raw_critic_response": "..."
}
```

```json
{
  "step_index": 3,
  "event": "react_fact_usage_summary",
  "total_facts": 17,
  "used_facts": 5,
  "remaining_facts": 12,
  "used_fact_ids": ["phase_1_reading_0", "phase_3_exploration_0", "..."]
}
```

---

## 8. Known Design Choices and Caveats

1. **Debugger candidate generation is sequential** (N separate LLM calls per step).
2. **Planner proposal is not directly parsed into tool calls**; instead it is converted into guidance and materialized by another debugger LLM call.
3. **Proposal critic is fail-open** on call/parse errors.
4. **Planner parser fallback** defaults to candidate 0 on malformed outputs.
5. **Full history text can be large** and may approach model context limits in long trajectories.
6. Agent package is registered via explicit import in triad runner; no change was made to global `openhands/agenthub/__init__.py` export list.

---

## 9. Session Change Log

### Session: 2025-03 — Planner Prompt Enhancement & History Dedup

All changes applied to the `oracle_triad_codeact_agent` module to improve planner prompt quality and materialization:

#### 9.1 Truncation Fix for Base Instruction Events

**File**: `oracle_triad_codeact_agent.py` — `_render_history_text_full()`

**Problem**: `_truncate_text()` with 6000-char limit was applied to system message and initial user message when rendering them in the SESSION INDEX. This clipped the OpenHands system prompt mid-sentence, causing the planner to lose important context about the debugger's capabilities and constraints.

**Fix**: Removed `_truncate_text()` wrapping for system message content and user message content in the SESSION INDEX section. The helper `_truncate_text()` is still defined and available for other uses (e.g., truncating individual observations in the event body).

#### 9.2 Tool Descriptions Passed to Planner

**Files**:
- `oracle_triad_codeact_agent.py` — new method `_build_tool_descriptions()`, updated `_init_components()`
- `oracle_planner.py` — `__init__`, `_render_prompt`, `from_env` all accept `tool_descriptions: str`
- `planner_select_or_propose.j2` — new "Available Tools" section

**Problem**: The planner had no direct knowledge of the debugger's available tools. It could only infer tool names by observing candidate `[TOOL CALL]` formats, leading to hallucinated or incorrect tool names in proposals.

**Fix**: `_build_tool_descriptions()` iterates over `self.tools` (list of `ChatCompletionToolParam` dicts) and builds a human-readable summary with tool name, description (truncated to 500 chars), and parameters (each description truncated to 200 chars). This string is passed through `OraclePlanner.from_env(tool_descriptions=...)` and rendered in the Jinja template under `## Available Tools (used by the Blinded Debugger)`.

#### 9.3 Structured Proposal Format (REASONING + TOOL CALL)

**File**: `planner_select_or_propose.j2`

**Problem**: Planner proposals were free-form text with no required structure. The debugger received the proposal as guidance but sometimes struggled to distinguish reasoning from action intent.

**Fix**: Added `## Proposal Format Requirements` section to the planner prompt specifying:
- **Part 1 — REASONING**: Thought process / analysis text for the debugger to express
- **Part 2 — TOOL CALL**: Concrete tool invocation in `[TOOL CALL] tool_name({...})` format

Also added an example well-formed proposal and updated the final output rules to require both parts.

#### 9.4 Improved Guidance Injection (`_inject_planner_guidance`)

**File**: `oracle_triad_codeact_agent.py` — `_inject_planner_guidance()`

**Problem**: The guidance injection message was a simple wrapper with the marker `[ORACLE PLANNER GUIDANCE - APPROVED BY BLINDED CRITIC]`. The debugger didn't know the guidance contained two distinct parts.

**Fix**: Updated the injected message to explicitly describe the two-part structure (REASONING + TOOL CALL), with instructions for the debugger to incorporate reasoning naturally and execute the suggested tool call.

#### 9.5 SystemMessageAction Deduplication

**File**: `oracle_triad_codeact_agent.py` — `_render_history_text_full()`

**Problem**: Event 0 (`SystemMessageAction`) was rendered in the SESSION INDEX (as "History Event 0 OPENHANDS SYSTEM MESSAGE") AND again in the event body loop, duplicating potentially thousands of chars.

**Fix**: Added `if isinstance(event, SystemMessageAction): continue` at the top of the event body loop (alongside the existing `MessageAction` skip for user messages).

#### 9.6 Verified Outputs

- Both modified `.py` files pass `python -c "import ast; ast.parse(open(f).read())"` check
- Jinja template renders correctly with `tool_descriptions` variable (output length ~4013 chars)
- Key prompt sections present: "Available Tools", tool entries like "execute_bash", "str_replace_editor", "Proposal Format Requirements", "REASONING", "TOOL CALL"

#### 9.7 Evidence from Eval Run (GLM-5-FP8 v0.61.0-oracle-triad)

Sample planner prompt inspected: `evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/OracleTriadCodeActAgent/GLM-5-FP8_maxiter_100_N_v0.61.0-oracle-triad/oracle_planner_prompts/getmoto__moto-7365/step_0007_attempt_00.txt`

**Observations from actual output**:
- Tool catalog rendered correctly in prompt: `execute_bash`, `think`, `finish`, `task_tracker`, `str_replace_editor` with parameters
- Planner produced well-formed proposal with both REASONING and `[TOOL CALL]` parts:
  ```
  "proposal_response": "The issue describes that `update_item` with an `ADD` expression... Let me examine the DynamoDB type handling code...\n\n[TOOL CALL] str_replace_editor({\"command\": \"view\", \"path\": \"/workspace/getmoto__moto__5.0/moto/dynamodb/models/dynamo_type.py\", \"security_risk\": \"LOW\"})"
  ```
- Planner correctly chose to propose (directing to source code) rather than selecting candidates (which all searched test files)
- Planner reasoning was non-leaky: mentioned `__add__` method logically derivable from the issue description

#### 9.8 Reasoning Injection Fix for Materialized Responses

**File**: `oracle_triad_codeact_agent.py` — `_materialize_planner_proposal()`, new helper `_extract_reasoning_from_proposal()`

**Problem**: When the debugger LLM materializes a planner proposal, it often produces a response with `content: []` (empty) and only `tool_calls`. The downstream `response_to_actions()` uses `content` as the "thought" text for the first action, so an empty content means the planner's reasoning is silently lost from the trajectory.

Example from GLM-5-FP8 eval (getmoto__moto-7365 step 7):
```json
{
  "content": [],
  "role": "assistant",
  "tool_calls": [{"function": {"name": "str_replace_editor", "arguments": "..."}}]
}
```
The planner carefully crafted reasoning text ("The issue describes that `update_item`...") but the debugger dropped it.

**Fix**: After `self.llm.completion()` returns in `_materialize_planner_proposal`, check if:
1. Response has `tool_calls` (LLM decided to call a tool), AND
2. `content` is empty/None/empty-list

If both true, extract the REASONING portion from the original `planner_proposal` text (everything before the first `[TOOL CALL]` marker) and set it as `msg.content`. This ensures `response_to_actions()` picks it up as the thought text for the action.

New helper: `_extract_reasoning_from_proposal(proposal)` splits on `[TOOL CALL]` marker:
- If marker found at position > 0: returns text before it (stripped)
- If marker at position 0: returns empty string (no reasoning to extract)
- If no marker: returns the entire proposal (assume it's all reasoning)

**Effect**: The materialized action will carry the planner's reasoning as its `thought`, making the trajectory more informative and matching how the debugger would naturally produce reasoning + tool call when acting on its own.

#### 9.9 Workflow Phase Enforcement for SFT Data Quality

**Files**:
- `planner_select_or_propose.j2` — new "SFT Data Quality & Workflow Phase Discipline" section
- `validate_oracle_proposal.j2` — new "Workflow Phase Enforcement" section + expanded red flags

**Problem**: The planner and critics were allowing logical jumps that skip required workflow phases. For example, at step 15 of getmoto__moto-7365, all three candidates AND the planner's own proposal jumped from Phase 4 (test creation / reproduction confirmed) directly to Phase 6 (`str_replace` code edit), completely skipping Phase 5 (Fix Analysis). The critic also failed to catch this. This produces trajectories where the model learns to skip analysis — poor SFT data.

**Root cause**: The existing constraints ("logically incremental with no jumps", "Ignores or contradicts required workflow phases") were too vague. Neither the planner nor the critic had explicit knowledge of the phase ordering or what constitutes adequate Phase 5 analysis.

**Fix — Planner prompt additions**:
- New section "SFT Data Quality & Workflow Phase Discipline" explaining that trajectories become SFT data and each step must be self-contained and instructive.
- Explicit listing of the 8 workflow phases from History Event 1 in order.
- Phase enforcement rules: NEVER propose/select Phase 6 without Phase 5 shown in history. When all candidates skip a phase, MUST propose the missing phase step.
- Definition of "adequate Phase 5": must state what the bug is, where it is, why code is wrong, and how to fix it (via `think` tool call or reasoning text).
- Preference rule: prefer candidates with `think` calls for analysis over candidates that silently jump.

**Fix — Critic prompt additions**:
- New "Workflow Phase Enforcement" section with explicit phase sequence.
- Concrete violation examples: Phase 6 without Phase 5, Phase 7 without Phase 6, skipping Phase 4.
- Explanation of why phase skips are harmful for SFT data quality.
- New red flags: "Phase 6 edit action proposed when Phase 5 analysis is missing", "Phase skip: proposes action belonging to Phase N+2 when Phase N not completed", "REASONING text does not explain the 'why'".

**Expected effect for step 15 scenario**: The planner should now propose a `think` tool call for Phase 5 analysis ("The `__add__` method in DynamoType uses `float()` instead of `Decimal()`, causing precision loss; the fix is to replace `float` with `Decimal` in both `__add__` and `__sub__` methods...") before any `str_replace` edit. If the planner still proposes an edit, the critic should reject it citing missing Phase 5.

#### 9.10 `--instance-ids` CLI Argument for Selective Evaluation

**File**: `run_infer_oracle_triad.py`

**Problem**: There was no way to run the oracle-triad evaluation on a specific subset of instances by ID. Users had to evaluate the entire dataset or use `--eval-n-limit` which only controls the count, not which instances are selected.

**Fix**: Added `--instance-ids` argument (`nargs='+'`, default `None`) to the argument parser. When provided, the loaded dataset is filtered to only include rows whose `instance_id` matches one of the given IDs. A warning is logged for any requested IDs not found in the dataset.

Usage:
```bash
python run_infer_oracle_triad.py ... --instance-ids django__django-12345 astropy__astropy-6789
```

When omitted, all instances are evaluated as before (no behavioral change).

### Session: 2025-03-24 — Structured React Facts Integration

All changes add structured react-fact-based guidance to the planner, with stateful tracking and precondition-based critic validation.

#### 9.11 React Facts JSON Format and Loading

**Files**:
- `run_infer_oracle_triad.py` — new `_load_react_facts()` function, `_write_oracle_context_file()` updated
- `oracle_triad_codeact_agent.py` — `_load_oracle_context()` returns `(context_text, react_facts_data)` tuple

**Problem**: The oracle planner received background knowledge as free-form markdown (`{instance_id}_analysis.md`). This was unstructured — the planner had no way to know which facts were relevant at which workflow phase, what preconditions they required, or what actions they recommended.

**Solution**: Added support for structured `{instance_id}_react_facts.json` files. Format:
```json
{
  "stages": [
    {
      "stage": "phase_3_exploration",
      "goal": "...",
      "facts": [
        {
          "fact": "Query.output_field property returns self.select[0].field...",
          "preconditions": ["Should be in phase_3_exploration or later.", "Must already have identified..."],
          "reasoning_action_observation": {
            "reasoning": "The crash depends on what output_field the Subquery resolves to...",
            "action": "read_file path=.../query.py, lines 225–250",
            "observation": "Query.output_field property at line 237..."
          }
        }
      ]
    }
  ]
}
```

The `_load_react_facts()` function loads this from `ORACLE_PREPROCESS_DIR/{instance_id}_react_facts.json` and passes it as `react_facts` in the oracle context payload JSON. `_load_oracle_context()` now returns a tuple `(context_text, react_facts_data)`, where `react_facts_data` is the raw parsed dict (or `None`).

#### 9.12 ReactFactTracker Class

**File**: `oracle_planner.py` — new class `ReactFactTracker`

**Purpose**: Manages structured react facts with per-step usage tracking.

**Design**:
- On init, parses `stages[].facts[]` and assigns unique IDs: `{stage}_{index}` (e.g., `phase_3_exploration_0`)
- Stores each fact's content, stage, goal, preconditions, recommended reasoning, and recommended action
- Maintains a `_used: dict[str, bool]` map — facts start unused, and are marked used after accepted planner decisions
- `render_available_facts_text()` renders only unused facts grouped by stage — used facts are omitted from subsequent prompts
- `get_preconditions_for_facts(fact_ids)` returns structured preconditions for given fact IDs, for passing to the critic

**Integration**: Created in `_init_components()` and passed to `OraclePlanner.from_env()`. The tracker lives on the `OraclePlanner` instance and persists across steps.

#### 9.13 Planner Prompt — Available Reference Facts Section

**File**: `planner_select_or_propose.j2`

**Additions**:
- New conditional section `## Available Reference Facts (PRIVATE — for proposal guidance only)` rendered when `has_react_facts` is true
- Each fact displayed with: `[fact_id]`, fact content, preconditions, recommended reasoning, recommended action
- Six usage rules: check preconditions, align with current stage, adapt don't copy verbatim, reference all used fact IDs, use aggressively, once used they disappear
- Output JSON schema now includes `"referenced_fact_ids": ["fact_id_1", "fact_id_2"]`
- New rule: when proposing, should reference at least one fact if applicable unused facts remain

#### 9.14 PlannerDecision — referenced_fact_ids Field

**File**: `oracle_planner.py` — `PlannerDecision` dataclass, `_parse_response_or_none()`

**Additions**:
- New field `referenced_fact_ids: list[str]` (default `[]`) on `PlannerDecision`
- Included in `to_dict()` serialization
- `_parse_response_or_none()` extracts `referenced_fact_ids` from the planner's JSON output and passes them through to the decision

#### 9.15 Fact Preconditions Passed to Critic

**Files**:
- `proposal_critic.py` — `validate()` and `_render_prompt()` accept `fact_preconditions: list[dict] | None`
- `validate_oracle_proposal.j2` — conditional `## Fact Preconditions` section

**Design**: When the planner references facts in a proposal, the orchestration agent extracts preconditions for those facts via `ReactFactTracker.get_preconditions_for_facts()`. These are passed to `OracleProposalCritic.validate(fact_preconditions=...)`.

The critic template renders a "Fact Preconditions" section (conditional on non-empty list) that:
- Explains each referenced fact's ID, stage, summary, and preconditions
- Instructs the critic to check whether preconditions are satisfied by the interaction history
- Adds rejection rule: "Unsatisfied fact preconditions indicate an unjustified knowledge jump"

When `fact_preconditions` is empty or `None`, the section is omitted entirely (backward compatible).

#### 9.16 Orchestration Loop — Fact Tracking Integration

**File**: `oracle_triad_codeact_agent.py` — `step()` method

**Changes to planner loop**:
1. After each planner decision, extract `referenced_fact_ids` from the decision
2. Get `fact_preconditions` via `ReactFactTracker.get_preconditions_for_facts(referenced_ids)`
3. Pass `fact_preconditions` to `OracleProposalCritic.validate()` when validating proposals
4. On accepted decision (candidate selected or proposal approved), call `ReactFactTracker.mark_facts_used(referenced_ids)` — facts are consumed and removed from future prompts
5. After the planner loop, log a `react_fact_usage_summary` event with total/used/remaining counts and used fact IDs

#### 9.17 Verified Outputs — django__django-12663 Run

Run: GLM-5-FP8, v0.61.0-oracle-triad, `princeton-nlp/SWE-bench_Verified` test split, instance `django__django-12663`

**React facts loaded**: 17 facts across 5 stages (phase_1_reading: 1, phase_3_exploration: 12, phase_4_test_creation: 1, phase_5_fix_analysis: 2, phase_6_fix_implementation: 1)

**Fact consumption progression**:
- Step 1: 1/17 used (phase_1_reading_0 — crash site identification)
- Step 8: 2/17 (+ phase_4_test_creation_0)
- Step 10–14: 3→7/17 (exploration facts consumed progressively)
- Step 16–18: 8→10/17 (more exploration facts)
- Step 20: 11/17 (+ phase_5_fix_analysis_0)
- Step 22: 13/17 (+ phase_6_fix_implementation_0)
- Steps 23–57: plateaued at 15/17 (2 remaining facts never consumed: phase_3_exploration_3, phase_3_exploration_10)

**Planner fact reference behavior**:
- 62 planner decisions total, 24 referenced at least one fact
- Facts referenced in both `candidate` and `proposal` decisions
- Multiple facts referenced in single decisions (e.g., step 14: `phase_3_exploration_4` + `phase_3_exploration_5`)

**Critic with preconditions**:
- 21 critic validations total
- Critic correctly rejected proposals with unsatisfied preconditions (e.g., step 5: rejected for phase ordering violation, step 8: rejected for skipping Phase 2, step 23: rejected for incomplete Phase 4)

**Remaining issues observed**:
- 2 of 17 facts were never consumed (phase_3_exploration_3, phase_3_exploration_10) — may need to investigate whether their preconditions were never met or the planner simply didn't pick them up
- Planner stopped referencing facts after step 37 even though 2 remained — the trajectory ran 57 steps total, suggesting the debugger got stuck in a verification loop
- `referenced_fact_ids` is sometimes referenced with stale IDs from already-used facts (e.g., step 8 attempt 0 references `phase_3_exploration_0` which was later consumed at step 10, but was referenced before being consumed — this is correct behavior since facts are only marked used on accepted decisions)

---

## 10. Suggested Next-Session Continuation Checklist

1. **Investigate unconsumed facts**: In the django__django-12663 run, 2/17 facts were never used (`phase_3_exploration_3`, `phase_3_exploration_10`). Check if their preconditions were impossible to satisfy given the trajectory, or if the planner needs prompt tuning to use them.
2. **Evaluate fact usage quality**: Review planner prompts to see whether fact reasoning/actions are being adapted well (vs. copied verbatim or ignored). Check if the recommended action paths match the actual workspace file layout.
3. **Test on more instances**: Run with additional react facts JSONs to verify generalization. Create `{instance_id}_react_facts.json` files for other SWE-bench instances.
4. **Fact precondition enforcement in critic**: Verify the critic is actually using preconditions to reject proposals with unsatisfied preconditions (not just rubber-stamping). Check critic prompt dumps for evidence.
5. **Late-trajectory stall**: The django run plateaued at 15/17 facts used from step 37 onwards (20 more steps with no new fact consumption). Investigate if the debugger got stuck in a verification/retry loop and whether fact-guided proposals could have helped break out.
6. **Fact-driven proposal vs. candidate selection**: Currently facts can be referenced in both `candidate` and `proposal` decisions. Consider whether fact references in `candidate` decisions should also mark facts as used (they currently do — verify this is desired).
7. Run materialization quality check: does the debugger faithfully execute fact-guided proposals?
8. Stress test with high `BLINDED_DEBUGGER_NUM_CANDIDATES` for latency/cost behavior.
9. Consider stricter fail behavior (or bounded fail-open) for proposal critic depending on reliability findings.

---

## 11. Minimal Run Example

```bash
cd /home/v-murongma/code/OpenHands_SWE-Bench-Optimized

bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
  llm.eval_qwen3_coder_30b_a3b_instruct HEAD OracleTriadCodeActAgent 1 100 1 \
  princeton-nlp/SWE-bench_Verified test 1
```

To use react facts, set the preprocess directory:

```bash
export ORACLE_PREPROCESS_DIR="/home/v-murongma/code/OpenHands_SWE-Bench-Optimized/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/preprocess"
```

The loader will automatically pick up `{instance_id}_react_facts.json` and `{instance_id}_analysis.md` from this directory.

Optional debug prompt capture:

```bash
export ORACLE_PLANNER_SAVE_PROMPTS=1
export ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS=1
```

Expected artifacts under eval output dir:

- `output.jsonl`
- `oracle_planner_context/<instance_id>.json`
- `oracle_triad_logs/<instance_id>.jsonl`
- `oracle_planner_prompts/<instance_id>/...` (if enabled)
- `oracle_proposal_critic_prompts/<instance_id>/...` (if enabled)

---

## 12. Run Result Template

Use the following template for every triad run so results are comparable across sessions.

### 12.1 Experiment Summary (copy/paste)

```markdown
## Oracle Triad Run - <RUN_ID>

- Date:
- Operator:
- Commit:
- Branch:
- Dataset:
- Split:
- Eval limit:
- Max iterations:
- Num workers:
- N runs:

### Model Config

- Debugger model config (`--llm-config`):
- Planner model config (`ORACLE_PLANNER_LLM_CONFIG`):
- Proposal critic model config (`ORACLE_PROPOSAL_CRITIC_LLM_CONFIG`):

### Triad Controls

- BLINDED_DEBUGGER_NUM_CANDIDATES:
- ORACLE_PLANNER_MAX_RETRIES:
- ORACLE_PLANNER_SAVE_PROMPTS:
- ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS:

### Output Paths

- eval_output_dir:
- output.jsonl:
- oracle_triad_logs/:
- oracle_planner_context/:
- oracle_planner_prompts/: (if enabled)
- oracle_proposal_critic_prompts/: (if enabled)
```

### 12.2 Aggregate Metrics (copy/paste)

```markdown
### Aggregate Metrics

- Total instances attempted:
- Total instances completed:
- Runtime (wall clock):
- Avg runtime per instance:

#### Planner/Critic Dynamics

- Total debugger candidates generated:
- Avg candidates per step:
- Planner decision counts:
  - candidate:
  - proposal:
- Proposal critic validations:
  - passed:
  - rejected:
- Planner retry distribution:
  - attempt=0:
  - attempt=1:
  - attempt=2+:
- Fallback-to-best-candidate count:
- Fallback rate (fallback / total steps):
```

### 12.3 Per-Instance Diagnostic Template

```markdown
### Instance - <INSTANCE_ID>

- Status: success | failed | timeout | aborted
- Final patch generated: yes/no
- Test result summary:
- Total agent steps:

#### Triad Trace Summary

- debugger_candidate events:
- oracle_planner_decision events:
- proposal_critic_validation events:
- planner proposals accepted:
- planner proposals rejected:
- final fallback used: yes/no

#### Notable Failures / Risks

- Leakage-like signals observed:
- Logic jump signals observed:
- Repeated critic rejection pattern:
- Any malformed planner JSON events:

#### Notes

-
```

### 12.4 Suggested Quick Commands

Use these commands to quickly populate key metrics from triad logs.

```bash
# Count event types for one instance
jq -r '.event' <oracle_triad_logs/INSTANCE.jsonl> | sort | uniq -c

# Planner decisions for one instance
jq -r 'select(.event=="oracle_planner_decision") | .decision' <oracle_triad_logs/INSTANCE.jsonl> | sort | uniq -c

# Proposal critic pass/fail for one instance
jq -r 'select(.event=="proposal_critic_validation") | .valid' <oracle_triad_logs/INSTANCE.jsonl> | sort | uniq -c

# Fallback approximator: planner retries exhausted warnings are not in JSONL,
# so estimate from last invalid proposal per step with no subsequent valid proposal.
```

### 12.5 Session Handoff Block (copy/paste)

```markdown
### Handoff for Next Session

- Main takeaway:
- Blocking issue (if any):
- Highest-priority next action:
- Files to inspect first:
  - evaluation/benchmarks/swe_bench_optimized/ORACLE_TRIAD_MODULE_REPORT.md
  - evaluation/benchmarks/swe_bench_optimized/run_infer_oracle_triad.py
  - openhands/agenthub/oracle_triad_codeact_agent/oracle_triad_codeact_agent.py
  - openhands/agenthub/oracle_triad_codeact_agent/oracle_planner.py (ReactFactTracker class)
- React facts JSON format reference:
  - evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/preprocess/django__django-12663_react_facts.json
- Representative logs:
  - <path to oracle_triad_logs/instance.jsonl>
```

---

## 14. Session: 2026-04-06 — Graph Fact System v5, DAG-Gated Availability, Investigation-First

### Overview

Major rewrite of the fact tracking system to support v5 `stage3_bridged.json` format, enforce DAG-gated availability with investigation-first ordering, simplify usage tracking to boolean, and remove overly strict workflow enforcement.

### Changes

#### 14.1 Three-Format Fact Loading
- `_load_react_facts()` now tries: `stage3_bridged.json` → `stage2_facts.json` → `_react_facts.json`
- Shell launcher auto-detects `swegym_v5` → `swegym_v3` → bare preprocess dir
- `ReactFactTracker` normalizes v5 single-dict evidence to 1-element list internally

#### 14.2 Simplified Boolean Usage Tracking
- Replaced per-evidence `_evidence_used[node_id] = [False, ...]` with `_used[node_id] = False/True`
- Duplicate marking silently skipped (was producing repeated log entries)
- `mark_facts_used(fact_ids, step_index=N)` now logs step number: `Marked f9 as used (step 15).`

#### 14.3 DAG-Gated Availability with Investigation Bypass
- Nodes only shown to planner when all precondition nodes are used
- Once all investigation nodes (`fact`, `bridge_fact`) are consumed, ALL remaining implementation nodes (`organizational_fact`, `plan_fact`, `edit_step`, `validation_step`) become available — preconditions bypassed

#### 14.4 Removed Strict Workflow Enforcement
- Removed A-family rules (A1–A4) from symbolic rule engine — caused false rejections (e.g. runtime probes rejected as "verification before implementation")
- Planner prompt: workflow is now "Recommended Workflow" (soft guideline)
- Critic prompt: removed "Workflow Phase Enforcement" section and all phase-specific red flags

#### 14.5 Natural Proposal Format & No Oracle Leakage
- Replaced "Part 1 — REASONING / Part 2 — TOOL CALL" with natural developer-style response
- Added hard constraint: **NEVER mention fact IDs, node IDs, or oracle concepts in `proposal_response`** — becomes SFT training data
- Encourages using fact `motivation` fields to craft natural explanations

#### 14.6 Fact Consumption Criteria
- Consume ONLY when: (1) fact's action performed AND (2) fact's statement visible in history
- If action done but statement not articulated → propose `think` call to state finding
- Selecting a candidate that approaches but doesn't fulfill → do NOT consume
- "Complete investigation before implementation": blocks code edits while investigation facts remain

#### 14.7 Verifier Improvements
- Added retry mechanism (2 retries) for LLM call failures + JSON parse failures in rule resolution
- Added D4 rule (`evidence.D4_fact_prerequisites_visible`): checks prerequisite facts' knowledge is visible in history (LLM-assisted, medium severity)
- Improved fact precondition rendering in critic/extraction templates (fixed empty `(stage: )` display)
- Rejected proposal text now included in planner feedback so it knows what was rejected

#### 14.8 Bug Fixes
- `best_candidate_index is None` TypeError crash — added None guard in `_plan_next_response`
- Duplicate fact marking — silently skip already-used nodes

---

## 15. Filled Session Handoff — 2026-04-06

### Handoff for Next Session

- **Main takeaway**: The oracle triad agent now supports v5 graph facts (`stage3_bridged.json`) with DAG-gated availability & investigation-first enforcement. Usage tracking simplified to boolean. Strict workflow rules removed. Verified working on MONAI-1012 instance — facts consumed progressively across steps, investigation completed before edit proposals.
- **Blocking issues**: 
  - GLM-5-FP8 occasionally produces non-JSON planner responses (handled by existing retry logic)
  - B4 rule can sometimes block fact-guided proposals that specify line ranges the debugger hasn't discovered via grep yet (LLM resolution helps but adds latency)
- **Highest-priority next actions**:
  1. Run multi-instance evaluation (e.g. 10 instances) to measure fact consumption rates and trajectory quality
  2. Check whether the "investigation-first" rule produces unnecessarily long trajectories when facts are redundant
  3. Evaluate whether D4 rule catches real prerequisite violations without too many false positives
- **Files to inspect first**:
  - `openhands/agenthub/oracle_triad_codeact_agent/oracle_planner.py` — `ReactFactTracker` (DAG, investigation bypass, boolean usage)
  - `openhands/agenthub/oracle_triad_codeact_agent/prompts/planner_select_or_propose.j2` — fact usage rules, constraints, proposal format
  - `openhands/agenthub/oracle_triad_codeact_agent/symbolic_rules.py` — A-family removed, D4 added
  - `openhands/agenthub/oracle_triad_codeact_agent/verifier.py` — retry mechanism, D4 rule integration
- **Representative logs**:
  - `evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/OracleTriadCodeActAgent/GLM-5-FP8_maxiter_100_N_v0.61.0-oracle-triad/` — MONAI-1012 instance
- **Key risks to watch**:
  - The "complete investigation before implementation" rule may force unnecessary investigation when the debugger has already figured out the fix
  - Long trajectories (60+ steps) suggest the agent sometimes gets stuck — may need a "give up on remaining facts" threshold
  - Fact-guided proposals specifying exact line ranges get rejected by B4 if the debugger hasn't grepped first — the planner must use grep to discover line numbers before proposing view ranges

---

## 16. Session: 2026-04-07 — Performance Optimizations

### 16.1 Windowed Planner History
- Planner now receives windowed history (SESSION SUMMARY + last N action steps) instead of full history
- Env var: `ORACLE_PLANNER_HISTORY_WINDOW=5` (default), `-1` or `0` for full history
- Critic/verifier still receives full history (needs it for grounding checks)
- New method: `_render_history_text_windowed(events, window=5)` alongside existing `_render_history_text_full(events)`
- Expected savings: ~5-15K tokens per planner call at later steps (history=7K→2K at step 24)

### 16.2 Batched LLM Rule Resolution (Stage 3.5)
- Replaced per-rule sequential LLM calls with a single batched call
- New template: `prompts/resolve_rules_batch.j2` — renders all failing rules + shared evidence in one prompt
- LLM returns `{"results": [{"rule_id": "B2", "verdict": "overruled", "reason": "..."}, ...]}` 
- Per-rule context fields (`oracle_dependent_claims`, `reasoning_claims`, `nearby_evidence`) preserved under each rule section
- Reduced 1-7 LLM calls to exactly 1 per verification attempt
- Old template `resolve_rule.j2` kept as reference but no longer called

### 16.3 Timing Instrumentation
- Per-step timing in agent log: `Step N timing: candidates=Xs (N calls), planner=Xs (N calls), verifier=Xs (N calls), materialization=Xs, total=Xs (N LLM calls)`
- Per-verifier-step timing: `Step N attempt M: extraction=Xs, retrieval=Xs, symbolic=Xs, llm_resolution=Xs, synthesis=Xs, total=Xs (N LLM calls) verdict=X`
- Planner prompt composition: `Step N prompt: ~XXXX tokens | issue=, oracle=, history=, candidates=, facts=, tools=, feedback=, template=`
- Timing events logged to triad JSONL as `step_timing` events
- Verifier exposes `_timing` metadata on `VerificationVerdict` for accurate LLM call counting

### 16.4 Bug Fixes
- Fixed verifier LLM call count always showing 0 in agent log (was using non-existent `_llm_calls` key)
- Fixed `ORACLE_PLANNER_HISTORY_WINDOW`: 0 and -1 now both mean full history (not clamped to 1)

---

## 17. Filled Session Handoff — 2026-04-07

### Handoff for Next Session

- **Main takeaway**: Major performance optimizations: windowed planner history (saves ~50% of history tokens at step 20+), batched LLM rule resolution (1 call instead of 1-7), comprehensive timing instrumentation. Tested on getmoto__moto-6857 — 52 steps with all 33 facts consumed, timing shows planner ~7s, verifier ~20-30s per proposal validation.
- **Blocking issues**: 
  - GLM-5-FP8 Non-JSON planner responses (10%+ rate in late steps when prompt >25K tokens) — may need to truncate oracle context or use a more instruction-following model
  - Verifier extraction stage takes 10-30s per call (LLM-based claim extraction) — `VERIFIER_PROGRAMMATIC_ONLY=1` can eliminate this but with less precise claims
- **Highest-priority next actions**:
  1. Run the fast config (`CANDIDATES=1, RETRIES=1, VALIDATOR=none`) on 10+ instances to measure resolve rate without verification overhead
  2. Evaluate windowed history impact on planner decision quality (does it miss important context from earlier steps?)
  3. Consider using programmatic-only extraction + batched resolution as the default "balanced" mode
  4. Add a mechanism to "give up" on remaining facts after N consecutive steps without fact consumption (prevents stalling at 50+ steps)
- **Files to inspect first**:
  - `openhands/agenthub/oracle_triad_codeact_agent/oracle_triad_codeact_agent.py` — windowed history, timing instrumentation
  - `openhands/agenthub/oracle_triad_codeact_agent/verifier.py` — batched resolution, timing
  - `openhands/agenthub/oracle_triad_codeact_agent/prompts/resolve_rules_batch.j2` — batched template
- **Representative timing data** (getmoto__moto-6857, window=full, 3rd run):
  - Step 1-10: ~8-35s per step (planner ~6s, verifier ~20s when triggered)
  - Step 20-30: ~10-130s per step (history growing, JSON parse retries)
  - Step 40-52: ~8-145s per step (large prompts, intermittent non-JSON)
- **Key configuration for next run**:
  ```bash
  export BLINDED_DEBUGGER_NUM_CANDIDATES=1
  export ORACLE_PLANNER_HISTORY_WINDOW=5
  export ORACLE_PLANNER_MAX_RETRIES=1
  export PROPOSAL_VALIDATOR=verifier  # or none for speed
  ```
