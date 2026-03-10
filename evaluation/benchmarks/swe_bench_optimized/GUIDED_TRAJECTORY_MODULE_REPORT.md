# Guided Trajectory Experiment — Module Report

_Workspace_: `/home/v-murongma/code/OpenHands_SWE-Bench-Optimized`  
_Python env_: `openhands-swebench` (conda)

---

## 1. Overview

The **Guided Trajectory** experiment tests whether an agent given a golden patch as a navigation hint still produces a *reachable* trajectory — one where every step can be justified by observed evidence.

The design is:

1. **GuidedCodeActAgent** receives the standard SWE-bench task *plus* a sealed `<REFERENCE_INFORMATION>` block containing the verified patch and test.
2. After every LLM call, a second independent LLM — the **Blinded Critic** — reviews the agent's response before it is accepted. The critic never sees the golden reference.
3. If the critic rejects the step (leakage or unjustified knowledge detected), the critique is fed back to the agent for up to `BLINDED_CRITIC_MAX_RETRIES` retries. If the agent still fails, the step is accepted anyway (with the failure logged) to prevent an infinite loop.
4. All critic decisions are written to a per-instance JSONL file for post-hoc analysis.

---

## 2. File Inventory

| File | Role |
|------|------|
| `openhands/agenthub/guided_codeact_agent/__init__.py` | Registers `GuidedCodeActAgent` with the `Agent` registry |
| `openhands/agenthub/guided_codeact_agent/guided_codeact_agent.py` | Agent implementation + per-process log helpers |
| `openhands/agenthub/guided_codeact_agent/blinded_critic.py` | Critic LLM wrapper + `ValidationResult` dataclass |
| `openhands/agenthub/guided_codeact_agent/prompts/validate_response.j2` | Critic prompt template |
| `evaluation/benchmarks/swe_bench_optimized/prompts/swe_guided.j2` | Agent task instruction template (includes golden reference block) |
| `evaluation/benchmarks/swe_bench_optimized/run_infer_guided.py` | Evaluation entry point — wraps `process_instance` |
| `evaluation/benchmarks/swe_bench_optimized/scripts/run_guided_infer.sh` | Shell launcher for the evaluation |

---

## 3. File-by-File Reference

### 3.1 `__init__.py`

```python
from openhands.agenthub.guided_codeact_agent.guided_codeact_agent import GuidedCodeActAgent
from openhands.controller.agent import Agent
Agent.register('GuidedCodeActAgent', GuidedCodeActAgent)
```

Must be imported before any code that resolves agent class strings. `run_infer_guided.py` does this explicitly near the top:
```python
import openhands.agenthub.guided_codeact_agent  # noqa: F401
```

---

### 3.2 `guided_codeact_agent.py`

**Class**: `GuidedCodeActAgent(CodeActAgent)` — `VERSION = '1.0'`

**Key attributes**:

| Attribute | Default | Source |
|-----------|---------|--------|
| `_blinded_critic` | `None` until first `step()` | lazily built via `BlindedCritic.from_env()` |
| `_max_retries` | `3` | `BLINDED_CRITIC_MAX_RETRIES` env var |
| `validation_log` | `[]` | accumulated in-memory, mirrored to `/tmp/blinded_critic_<PID>.jsonl` |

**Control flow of `step(state)`**:

```
1. Handle pending_actions / /exit early exits (same as CodeActAgent)
2. Condense history events  
3. Lazily init BlindedCritic from issue text (golden block stripped)
4. Build LLM messages
5. VALIDATION LOOP (up to _max_retries+1 attempts):
   a. Call primary LLM → response
   b. If no critic: break immediately
   c. Extract response text (_extract_response_text)
   d. critic.validate(step_index, history_text, response_text)
   e. Append entry to validation_log + write to /tmp/blinded_critic_<PID>.jsonl
   f. If valid: break
   g. If attempt < max_retries: inject critique into messages, retry
   h. If attempt == max_retries: log warning, accept anyway
6. Parse response → actions → queue → return first action
```

**Step index**: `step_index = state.iteration_flag.current_value`
> ⚠️ Do NOT use `state.iteration` (deprecated, returns `None`) or `state.get_local_step()` (subtracts `parent_iteration=100`, giving `-99, -98, …`).

**Per-process log helpers** (module-level, importable by `run_infer_guided.py`):

```python
get_validation_log_path()       # → /tmp/blinded_critic_<PID>.jsonl
clear_validation_log()          # delete the file
read_and_clear_validation_log() # → list[dict], then delete file
_append_validation_entry(entry) # append one JSON line (internal use)
```

**`_extract_response_text(response)`**: converts litellm `ModelResponse` to a human-readable string for the critic. Includes:
- Textual thought / reasoning from `msg.content`
- Tool calls formatted as `[TOOL CALL] fn_name(full_arguments_json)`

> ⚠️ Arguments are passed in full — do NOT add a `[:400]` slice. The critic needs the complete content to detect leakage.

**`_inject_critique(messages, feedback)`**: appends a `role='user'` `Message` containing the critic's rejection reason + repair instructions + `"Please provide a revised response…"`.

**`_strip_reference_block(text)`**: removes `<REFERENCE_INFORMATION>…</REFERENCE_INFORMATION>` (including its contents) from the task instruction before passing it to the critic as `issue_text`.

**`_render_history_text(events)`**: converts condensed event list to a plain-text summary for the critic.

Two-pass design:
1. **Pass 1 — session index**: scans all events and builds a compact header listing every unique file read, every file edited, and the count of commands run. This preamble survives condenser truncation and lets the critic see at a glance what the agent has already accessed.
2. **Pass 2 — event log**: emits one line per event, labelled `[Event N]` (where N is the raw enumerate index). **Important**: these are NOT step indexes — they are sequential event IDs. Using `[Event N]` avoids confusion with the iteration-based `step_index` in the prompt header.

Truncation limits (all were previously 500 chars):

| Event type | Old limit | New limit |
|---|---|---|
| `FileReadObservation` content | 500 | 3 000 (with total-char annotation) |
| `CmdOutputObservation` content | 500 | 2 000 |
| `CmdRunAction` command | 300 | 600 |
| `FileEditAction` content | 200 | 400 |
| Agent `MessageAction` | 500 | 800 |
| `AgentThinkAction` thought | 300 | 400 |

> ⚠️ History is passed through the condenser before `_render_history_text` sees it. The session index in Pass 1 is the only way to confirm which files were read in earlier steps that have been condensed away.

---

### 3.3 `blinded_critic.py`

**Dataclass `ValidationResult`**:

| Field | Type | Notes |
|-------|------|-------|
| `step_index` | `int` | 0-based, same as `state.iteration_flag.current_value` |
| `valid` | `bool` | True = step accepted |
| `reason` | `str` | one-sentence justification |
| `unjustified_knowledge` | `list[str]` | items known without prior observation |
| `prerequisite_conditions` | `list[str]` | steps needed before this action is justified |
| `feedback_message` | `str` | formatted rejection message sent back to agent |
| `agent_response_text` | `str` | full response string that was judged |
| `raw_critic_response` | `str` | raw LLM output for debugging |

`to_dict()` serialises all fields for JSONL logging.

**Class `BlindedCritic`**:

Constructor: `__init__(llm: LLM, issue_text: str)`  
`PROMPTS_DIR`: sibling `prompts/` directory.

**`validate(step_index, history_text, agent_response_text, attempt=0) → ValidationResult`**:
1. Render `validate_response.j2` with the four variables.
2. Call `self.llm.completion(messages=[{'role': 'user', 'content': prompt}])`.
3. Call `_maybe_save_prompt(step_index, attempt, prompt, raw_response)` (saves to disk if enabled; see below).
4. If call fails, return a `valid=True` fallback (fail-open to avoid blocking the agent).
5. Parse via `_parse_response`.
6. Attach `agent_response_text` to the result.

The `attempt` parameter is 0-based and is passed from the agent's retry loop so that debug files are named per attempt.

**`_maybe_save_prompt(step_index, attempt, prompt, raw_response) → None`** (static):
- No-op unless `BLINDED_CRITIC_SAVE_PROMPTS_DIR` env var is set.
- If set, creates the directory and writes `step_{N:04d}_attempt_{M:02d}.txt` containing the full rendered prompt followed by the raw LLM response.
- Called on both success and LLM-call failure.
- `BLINDED_CRITIC_SAVE_PROMPTS_DIR` is set per-instance by `run_infer_guided.py` (see §3.6). Never set it manually across workers — they would collide.

**`_parse_response(step_index, raw_text) → ValidationResult`**:
1. Extract JSON with `_extract_json` (handles ` ```json ``` ` fences or bare `{…}`).
2. Fail-open if JSON absent or malformed.
3. **Enforced invariant** — non-empty `unjustified_knowledge` forces `valid = False` regardless of what the LLM returned:
   ```python
   if unjustified:
       valid = False
   else:
       valid = bool(data.get('valid', True))
   ```
   > This guards against the LLM writing `"valid": true` with a non-empty `unjustified_knowledge` list, which was a real observed failure mode.

**`_build_feedback_message(…) → str`**: generates the rejection notice sent to the agent:
```
[QA REVIEW — ACTION REJECTED]

Reason: <critic's reason>

Information used that is NOT yet derivable from observations:
  - <items>

Prerequisite steps / conditions:
  - <items>

IMPORTANT — How to fix your response:
  1. Do NOT mention the reference patch, reference test, or any information from them.
  2. You MAY reference names that appear in the issue description (class names,
     function names, error messages). What you must NOT do is assume their exact
     file path, import path, or implementation details without first exploring.
     Example: "I will look for separability_matrix" is fine; writing
     "from astropy.modeling.separable import separability_matrix" without having
     read any source file is NOT fine.
  3. Your NEXT action must be a concrete exploration tool call…
  4. After the tool call returns, reason from THAT output — not from prior knowledge.
  5. Keep your thought text minimal: state only what you are about to do and why
     it follows from prior observations. Do NOT pre-announce specific import paths,
     exact values, or fixes before reading the code.

In short: explore first, then conclude. Every claim about implementation details
must cite a tool output from this session as its evidence.
```

> **Why rule 2 was rewritten**: the previous rule 2 said "Do NOT state any specific function name", which was too broad — it caused the agent to fail retries even when naming symbols that appeared in the issue description. The new rule draws a precise line: issue-mentioned names are allowed; their implementation location and internal details are not.

**`from_env(issue_text) → BlindedCritic | None`** (classmethod):
- Reads LLM config named by `BLINDED_CRITIC_LLM_CONFIG` env var (default: `blinded_critic`) from the file in `CONFIG_FILE` env var (default: `config.toml`).
- Calls `get_llm_config_arg(config_name, config_file)`.
- Sets `llm_config.log_completions = False` to suppress critic records from cluttering LLM completion logs.
- Returns `None` (validation disabled) if config section is absent — agent continues normally.

---

### 3.4 `prompts/validate_response.j2`

**Template variables**: `issue_text`, `step_index`, `history_text`, `agent_response_text`

**Structure**:
1. Role definition: "Blinded Critic, independent QA reviewer"
2. Two properties enforced: **Reachability** and **Non-leakage**
3. **"Important Context: The Agent Had a Golden Reference Block"** — tells the critic the agent received a hidden `<REFERENCE_INFORMATION>` block. Lists **auto-invalid red-flag phrases**:
   - "looking at the reference patch", "the reference patch shows/says", "according to the patch", "the golden patch", "as shown in the reference", "the patch mentions", "as specified in the patch"
   - Citing an exact fix without a prior read tool call
   - Stating the exact buggy value + replacement without prior source observation
   - "found the exact line mentioned in the reference patch"
4. Issue description section
5. **Accumulated interaction history section** — header reads "before Step *N*" (not a range). Begins with two important notes:
   - **Truncation note**: if a file appears in the "Files READ this session" index, the agent has seen its full content even if the log shows only a preview — the index is authoritative.
   - **Numbering note**: events in the log are labelled `[Event N]` (sequential position in the condensed event list). This is distinct from the iteration-based `Step N` in the prompt header.
6. Agent's proposed response section
7. Evaluation task (leniency rules — updated in this session):
   - Scan red flags **first**
   - **Names in the issue description are NOT unjustified** — the agent may freely use any class/function/module name that appears verbatim in the issue text
   - **What IS unjustified**: exact import paths, exact file paths, exact line numbers, exact code values not yet observed via a tool call
   - Lenient for general software engineering knowledge; strict for project-specific implementation details
   - **Never lenient** for specific fix details (exact old value + replacement) without a prior read
8. **Output format** — single JSON object:
   ```json
   {"valid": true|false, "reason": "…", "unjustified_knowledge": [], "prerequisite_conditions": []}
   ```
9. **CRITICAL RULE** explained inline: `unjustified_knowledge` non-empty → `"valid": false`. Self-contradictory responses are explicitly warned against.

---

### 3.5 `prompts/swe_guided.j2`

Extends the standard `swe_gpt4.j2` 8-phase workflow. The key addition is the `<REFERENCE_INFORMATION>` block injected after the issue description, populated with `{{ instance.patch }}` and `{{ instance.test_patch }}`.

The block was **completely rewritten** in this session. The old framing ("navigation aid so you don't waste iterations") was counterproductive — it told the agent it was acceptable to shortcut exploration. The new framing leads with the actual goal.

**REFERENCE_INFORMATION block (current agent instructions)**:

```
The following verified patch and test are provided so you know exactly where this
task ends up. Your job is NOT to reproduce the patch mechanically — it is to
produce a realistic, step-by-step trajectory that a skilled engineer would follow
if they had to discover and fix this bug from scratch, starting only from the
issue description above.

YOUR GOAL: A REACHABLE TRAJECTORY
- Every file path, function name, variable name, or code value you mention must
  have appeared in a prior tool call output OR be explicitly named in the issue.
- Every conclusion must cite specific evidence from your own tool observations.
- You must actually run the exploration steps rather than skipping them.

Think of it this way: you are producing training data for a future agent that
will NOT have the reference. Every step must be reachable from what your tool
calls revealed.

RULES — A SECOND REVIEWER IS WATCHING

ALLOWED:
- Names (classes, functions, error messages) that appear verbatim in the issue
- Navigating to the relevant file/module, then reading it with a tool call
- Reproducing the failing behaviour described in the issue
- Reasoning from what your tool calls actually returned

FORBIDDEN:
- Referencing the patch or test explicitly
- Stating code values, variable names, exact line numbers not yet observed
- Writing a reproduction script with exact import paths from the patch before
  reading the source file
- Claiming to have "found" the bug before reading the code with a tool call
- Skipping exploration phases because you already know the answer

HOW TO PRODUCE A VALID STEP:
1. State what you observed and what you want to find out — cite prior tool output.
2. Issue a single focused tool call.
3. In the next step, reason from its output, not from prior knowledge.
4. When stating a fix, every detail must have appeared in YOUR tool output.
```

> **Why this framing**: "training data" is the most powerful motivator for the agent to follow evidence — it reframes the constraint as a goal rather than a restriction.  The old "navigation aid" framing actively encouraged skipping exploration.

---

### 3.6 `run_infer_guided.py`

Entry point for the guided evaluation. Run directly (not imported).

**Key differences from `run_infer.py`**:
- Default instruction template: `swe_guided.j2` (set via `os.environ['INSTRUCTION_TEMPLATE_NAME']` before anything else, unless already overridden)
- Default agent class: `GuidedCodeActAgent`
- `process_instance_guided` wraps `process_instance`:
  1. `clear_validation_log()` — clear any leftover log from a previous instance in this worker
  2. `process_instance(...)` — runs the agent
  3. `read_and_clear_validation_log()` — collect all critic decisions
  4. Write per-instance JSONL to `{eval_output_dir}/blinded_critic_logs/{instance_id}.jsonl`
  5. Attach `validation_log` to `output.test_result['validation_log']`
- `AGENT_CLS_TO_FAKE_USER_RESPONSE_FN['GuidedCodeActAgent']` patched to reuse the `CodeActAgent` handler (required or `process_instance` raises `KeyError`)

**Multi-run / batching**: mirrors `run_infer.py` — supports `--n-runs`, `SKIP_RUNS` env var, batching by `eval_num_workers`.

**Important imports at module top** (order matters):
```python
import openhands.agenthub                           # registers built-in agents
import openhands.agenthub.guided_codeact_agent      # registers GuidedCodeActAgent
```

---

### 3.7 `scripts/run_guided_infer.sh`

**Usage**:
```bash
bash evaluation/benchmarks/swe_bench_optimized/scripts/run_guided_infer.sh \
  <MODEL_CONFIG> <COMMIT_HASH> [AGENT] [EVAL_LIMIT] [MAX_ITER] \
  [NUM_WORKERS] [DATASET] [SPLIT] [N_RUNS]
```

**Positional args** (all after `MODEL_CONFIG` and `COMMIT_HASH` are optional):

| # | Arg | Default |
|---|-----|---------|
| 3 | AGENT | `GuidedCodeActAgent` |
| 4 | EVAL_LIMIT | (none — all instances) |
| 5 | MAX_ITER | `100` |
| 6 | NUM_WORKERS | `1` |
| 7 | DATASET | `princeton-nlp/SWE-bench_Lite` |
| 8 | SPLIT | `test` |
| 9 | N_RUNS | `1` |

**Extra env vars**:

| Env var | Default | Purpose |
|---------|---------|---------|
| `BLINDED_CRITIC_LLM_CONFIG` | `blinded_critic` | `[llm.<name>]` section in `config.toml` for the critic |
| `BLINDED_CRITIC_MAX_RETRIES` | `3` | max retries per step on rejection |
| `BLINDED_CRITIC_SAVE_PROMPTS` | `0` | Set to `1` to save every critic prompt + raw response to disk |
| `INSTRUCTION_TEMPLATE_NAME` | `swe_guided.j2` | Jinja2 template for task instruction |

When `BLINDED_CRITIC_SAVE_PROMPTS=1`, `run_infer_guided.py` automatically sets `BLINDED_CRITIC_SAVE_PROMPTS_DIR` to `{eval_output_dir}/blinded_critic_prompts/{instance_id}/` before each instance and clears it afterwards. Files are named `step_{N:04d}_attempt_{M:02d}.txt`.

**config.toml requirement**:
```toml
[llm.blinded_critic]
model = "gpt-4o-mini"       # or any model
api_key = "..."
temperature = 0.0
max_output_tokens = 1024
```

---

## 4. Data Flow Diagram

```
run_guided_infer.sh
  └─ run_infer_guided.py (process_instance_guided)
       │
       ├─ clear_validation_log()   ← wipe /tmp/blinded_critic_<PID>.jsonl
       │
       ├─ process_instance(...)
       │    └─ run_controller(GuidedCodeActAgent)
       │         └─ agent.step(state)  [called once per iteration]
       │              ├─ [init] BlindedCritic.from_env(issue_text)
       │              │                             ← strips <REFERENCE_INFORMATION>
       │              ├─ primary LLM call
       │              ├─ BlindedCritic.validate(step, history, response)
       │              │    ├─ render validate_response.j2
       │              │    └─ parse JSON → ValidationResult
       │              │         └─ enforce: unjustified_knowledge≠[] ⟹ valid=False
       │              ├─ _append_validation_entry → /tmp/blinded_critic_<PID>.jsonl
       │              ├─ if invalid & attempts_left: inject critique, retry
       │              └─ return action
       │
       ├─ read_and_clear_validation_log()
       │
       ├─ write blinded_critic_logs/{instance_id}.jsonl
       │
       └─ attach validation_log to output.test_result
```

---

## 5. Output Artifacts

After a run, for each instance `{ID}`:

| Path | Content |
|------|---------|
| `{eval_output_dir}/output.jsonl` | Standard EvalOutput, with `test_result.validation_log` array |
| `{eval_output_dir}/blinded_critic_logs/{ID}.jsonl` | Per-instance critic log — one JSON line per validation call |
| `{eval_output_dir}/blinded_critic_prompts/{ID}/step_NNNN_attempt_MM.txt` | Full critic prompt + raw response (only when `BLINDED_CRITIC_SAVE_PROMPTS=1`) |
| `{eval_output_dir}/llm_completions/{ID}/*.json` | Standard LLM completion records (critic completions excluded) |

**Per-entry schema in `blinded_critic_logs/{ID}.jsonl`**:
```json
{
  "step_index": 3,
  "attempt": 0,
  "valid": false,
  "reason": "Agent cites specific function name not yet observed.",
  "unjustified_knowledge": ["knows bug is in foo.bar() without reading the file"],
  "prerequisite_conditions": ["must read foo.py first"],
  "feedback_message": "...",
  "agent_response_text": "...",
  "raw_critic_response": "..."
}
```

---

## 6. Known Bugs Fixed

| Bug | Root cause | Fix |
|-----|-----------|-----|
| `KeyError: 'GuidedCodeActAgent'` | Not in `AGENT_CLS_TO_FAKE_USER_RESPONSE_FN` | Patched in `run_infer_guided.py` |
| `TypeError: '>' not supported between NoneType and int` | `state.iteration` is deprecated → `None` | Changed to `state.iteration_flag.current_value` |
| Steps showed `-99, -98, …` | `state.get_local_step()` subtracts `parent_iteration=100` | Same fix — use `iteration_flag.current_value` |
| Agent tool args truncated to 400 chars before critic | `fn.arguments[:400]` in `_extract_response_text` | Removed the slice — full string passed |
| All judgments valid despite clear leakage | `valid` evaluated independently of `unjustified_knowledge` | Python invariant enforced in `_parse_response` |
| Critic missed explicit oracle phrases | Critic had no context that agent had a golden block | Added "Important Context" + red-flag list to `validate_response.j2` |
| Agent repeatedly violated rules after rejection | Feedback message was too generic ("base your response on evidence") | Rewritten as numbered repair instructions with explicit rule 2 distinction |
| Agent leaked at step 0 on every instance | `<REFERENCE_INFORMATION>` block only said "guidance only, don't copy" | Block rewritten with training-data framing + ALLOWED/FORBIDDEN/HOW TO rules |
| Critic false-positive: rejecting issue-mentioned names | Leniency rule said "don't flag specific function names" but was too broad — flagged `separability_matrix` even though it appeared in the issue | Added explicit carve-out: "names in the issue description are NOT unjustified knowledge" to both `validate_response.j2` and feedback rule 2 |
| Critic false-positive: "agent hasn't read the file" at step 50+ | `FileReadObservation` was truncated to 500 chars — critic couldn't confirm the file was read | Rewrote `_render_history_text`: session index preamble lists every file ever read; `FileReadObservation` limit raised to 3 000 chars; all other limits raised proportionally |
| Critic history said `[Step 6]` but prompt said "Step 2" | History labels used enumerate index `i`; prompt used `state.iteration_flag.current_value` — different counters | Renamed history labels to `[Event N]`; added note to critic prompt explaining the distinction |

---

## 7. Configuration Checklist for a New Run

1. **`config.toml`** — add `[llm.blinded_critic]` section (any model; temperature 0 recommended).
2. **Main LLM** — must be in a named `[llm.<name>]` section, passed as `MODEL_CONFIG` to the script.
3. **Conda env** — activate `openhands-swebench` before running.
4. **Launch command** (example — 1 instance, Verified test set):
   ```bash
   cd /home/v-murongma/code/OpenHands_SWE-Bench-Optimized
   conda run -n openhands-swebench \
   bash evaluation/benchmarks/swe_bench_optimized/scripts/run_guided_infer.sh \
     llm.eval_qwen3_coder_30b_a3b_instruct HEAD GuidedCodeActAgent 1 100 1 \
     princeton-nlp/SWE-bench_Verified test 1
   ```
5. **To save critic prompts for debugging** (add before the command above):
   ```bash
   export BLINDED_CRITIC_SAVE_PROMPTS=1
   ```
   Prompts land in `{eval_output_dir}/blinded_critic_prompts/{instance_id}/step_NNNN_attempt_MM.txt`.
6. **Inspect results**:
   ```bash
   # Per-instance critic log
   cat {eval_output_dir}/blinded_critic_logs/<instance_id>.jsonl | python -m json.tool
   
   # Quick stats — how many rejections?
   jq -r '.valid' {eval_output_dir}/blinded_critic_logs/<instance_id>.jsonl | sort | uniq -c
   
   # Inspect a specific critic prompt + response
   cat {eval_output_dir}/blinded_critic_prompts/<instance_id>/step_0006_attempt_00.txt
   ```

---

## 8. Pending / Future Work

- **Run a full eval** to verify the latest round of fixes (training-data framing in `swe_guided.j2`, issue-name carve-out, session index in history, raised truncation limits) actually reduce max-retry exhaustion rates in practice.
- **Aggregate leakage statistics**: post-process `blinded_critic_logs/` across all instances — rejection rate per step index, common unjustified-knowledge types, retry distribution. A script or notebook cell in `evaluation/evaluation_analysis/inspect_response.ipynb` would be the natural home.
- **Tune `BLINDED_CRITIC_MAX_RETRIES`** (currently 3) based on observed retry distributions from the next eval run.
- **Critic model selection**: evaluate whether `gpt-4o-mini` gives sufficient precision at lower cost, or whether a stronger model is needed for critic accuracy.
- **History budget vs. condenser truncation**: at step 50+ the condensed history may be short even with raised limits. Consider giving the critic access to the full LLM completion log as a fallback, or caching a per-step "files confirmed read" set that survives condensation.
- **`str_replace_editor` view tool**: these tool calls produce `FileReadObservation`-style output but are not typed as `FileReadAction` — they will not appear in the session index. Verify whether the agent predominantly uses `execute_bash cat` or `str_replace_editor view` and add a handler in `_render_history_text` if needed.
