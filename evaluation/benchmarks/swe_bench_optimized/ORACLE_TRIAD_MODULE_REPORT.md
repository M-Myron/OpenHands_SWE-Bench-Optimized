# Oracle Triad Experiment - Module Report

_Workspace_: `/home/chentianyu/murong/code/OpenHands_SWE-Bench-Optimized`  
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
| `openhands/agenthub/oracle_triad_codeact_agent/oracle_planner.py` | Oracle planner LLM wrapper + `PlannerDecision` parsing |
| `openhands/agenthub/oracle_triad_codeact_agent/proposal_critic.py` | Blinded proposal critic wrapper + `ProposalValidationResult` |
| `openhands/agenthub/oracle_triad_codeact_agent/prompts/planner_select_or_propose.j2` | Planner prompt template |
| `openhands/agenthub/oracle_triad_codeact_agent/prompts/validate_oracle_proposal.j2` | Proposal critic prompt template |
| `evaluation/benchmarks/swe_bench_optimized/run_infer_oracle_triad.py` | Eval entrypoint + per-instance oracle context writer |
| `evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh` | Shell launcher |

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
   - planner returns decision (`candidate` or `proposal`) plus `best_candidate_index`.
   - if `candidate`: choose selected/best candidate and continue.
   - if `proposal`:
     - if critic disabled, materialize proposal directly;
     - else validate proposal with blinded critic.
     - if critic passes: materialize proposal.
     - if critic fails and retries remain: send feedback back to planner.
     - if retries exhausted: fallback to planner `best_candidate_index`.
6. Convert chosen response to actions and return first queued action.

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

- `from_env(issue_text, oracle_context, tool_descriptions='')`
- config key: `ORACLE_PLANNER_LLM_CONFIG` (default `oracle_planner`)
- disables completion logging (`llm_config.log_completions = False`)
- `tool_descriptions` is passed to `_render_prompt` and rendered in the Jinja template "Available Tools" section

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
  "proposal_response": "..."
}
```

Important hard rules in prompt:

- planner must **always** output `best_candidate_index`, even when proposing.
- `proposal_response` MUST contain both REASONING text and a `[TOOL CALL]` suggestion using exact tool names from the Available Tools section.
- **SFT Data Quality & Workflow Phase Discipline**: explicit 8-phase workflow ordering enforced. NEVER propose/select Phase 6 (edit) without Phase 5 (analysis) shown in history. When all candidates skip a phase, MUST propose the missing phase step.

### 4.2 Proposal critic prompt

File: `validate_oracle_proposal.j2`

Checks:

- consistency with history
- no logic jumps
- no leakage/oracle mention
- no implementation details absent from history/issue
- **workflow phase ordering**: rejects proposals that skip required phases (e.g., Phase 6 edit without Phase 5 analysis in history).
- **SFT reasoning quality**: rejects proposals whose REASONING text doesn't explain "why"

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
  "raw_planner_response": "..."
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

---

## 10. Suggested Next-Session Continuation Checklist

1. Run one smoke eval (`EVAL_LIMIT=1`) with the latest code and verify:
   - oracle context file generated,
   - triad logs populated,
   - planner/critic retries visible when expected,
   - **new**: tool descriptions appear in planner prompts,
   - **new**: proposals contain both REASONING + TOOL CALL parts.
2. Investigate materialization quality: does the debugger faithfully execute the planner's suggested tool call when the two-part guidance is injected?
   - Check if the debugger ignores the `[TOOL CALL]` suggestion and does something else.
   - Check if the debugger duplicates reasoning or adds unnecessary preamble.
3. Add aggregate analysis script/notebook for `oracle_triad_logs/`:
   - proposal rate,
   - critic rejection rate,
   - fallback-to-best-candidate rate,
   - average planner attempts per step.
4. Stress test with high `BLINDED_DEBUGGER_NUM_CANDIDATES` for latency/cost behavior.
5. Evaluate whether planner guidance materialization should be replaced by direct structured action output to reduce one extra debugger call.
6. Consider stricter fail behavior (or bounded fail-open) for proposal critic depending on reliability findings.
7. Tune truncation limits in `_build_tool_descriptions()` (currently 500 chars for tool desc, 200 chars per param desc) — review whether longer descriptions help or hurt planner decision quality.

---

## 11. Minimal Run Example

```bash
cd /home/chentianyu/murong/code/OpenHands_SWE-Bench-Optimized

bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
  llm.eval_qwen3_coder_30b_a3b_instruct HEAD OracleTriadCodeActAgent 1 100 1 \
  princeton-nlp/SWE-bench_Verified test 1
```

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
- Representative logs:
  - <path to oracle_triad_logs/instance.jsonl>
```

---

## 13. Filled Session Handoff — 2025-03 Planner Prompt Enhancement

### Handoff for Next Session

- **Main takeaway**: Planner prompt was enhanced with tool catalog, structured REASONING+TOOL CALL proposal format, improved guidance injection, history dedup, and reasoning injection into materialized responses. Initial evidence from GLM-5-FP8 eval shows planner correctly using the format but the debugger was dropping reasoning text (content=[]) during materialization — now fixed by extracting reasoning from the proposal and injecting it into the response. No regressions observed in prompt rendering.
- **Blocking issue**: None. All changes compile and template renders correctly.
- **Highest-priority next action**: Run a smoke eval (`EVAL_LIMIT=1`) with the updated code to verify end-to-end materialization quality — specifically whether the debugger faithfully executes the planner's two-part guidance.
- **Files to inspect first**:
  - `evaluation/benchmarks/swe_bench_optimized/ORACLE_TRIAD_MODULE_REPORT.md` (this file, Section 9)
  - `openhands/agenthub/oracle_triad_codeact_agent/oracle_triad_codeact_agent.py` (changes in `_build_tool_descriptions`, `_inject_planner_guidance`, `_render_history_text_full`)
  - `openhands/agenthub/oracle_triad_codeact_agent/oracle_planner.py` (`tool_descriptions` param)
  - `openhands/agenthub/oracle_triad_codeact_agent/prompts/planner_select_or_propose.j2` (tool catalog + proposal format sections)
- **Representative logs**:
  - `evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/OracleTriadCodeActAgent/GLM-5-FP8_maxiter_100_N_v0.61.0-oracle-triad/oracle_planner_prompts/getmoto__moto-7365/step_0007_attempt_00.txt`
- **Key risks to watch**:
  - Debugger may ignore the `[TOOL CALL]` suggestion in guidance and generate a different action
  - Tool descriptions add ~2-3K tokens to every planner prompt — monitor for context window pressure in long trajectories
  - `_truncate_text()` removal for base instructions means the full system prompt (~4K chars) is included in every planner call
