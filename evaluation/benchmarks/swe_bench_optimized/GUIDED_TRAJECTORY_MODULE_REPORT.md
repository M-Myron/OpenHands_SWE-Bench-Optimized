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

**`_render_history_text(events)`**: converts condensed event list to a plain-text summary for the critic. Handles `CmdRunAction`, `FileReadAction`, `FileEditAction`, `CmdOutputObservation`, `FileReadObservation`, `FileEditObservation`, generic `Observation`. The initial user `MessageAction` is skipped (too long; critic already has the issue text).

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

**`validate(step_index, history_text, agent_response_text) → ValidationResult`**:
1. Render `validate_response.j2` with the four variables.
2. Call `self.llm.completion(messages=[{'role': 'user', 'content': prompt}])`.
3. If call fails, return a `valid=True` fallback (fail-open to avoid blocking the agent).
4. Parse via `_parse_response`.
5. Attach `agent_response_text` to the result.

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
  1. Do NOT mention the reference patch, reference test, or any information…
  2. Do NOT state any specific file name, function name, variable name…
  3. Your NEXT action must be a concrete exploration tool call…
  4. After the tool call returns, reason from THAT output — not from prior knowledge.

In short: explore first, then conclude. Every claim must cite a tool output…
```

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
5. Accumulated interaction history section (with step-range label)
6. Agent's proposed response section
7. Evaluation task:
   - Scan red flags **first**
   - Check referencing unobserved file/function/variable/code
   - Lenient for general software engineering knowledge; strict for project-specific details
   - **Not lenient** for specific fix details (exact wrong value + replacement)
8. **Output format** — single JSON object:
   ```json
   {"valid": true|false, "reason": "…", "unjustified_knowledge": [], "prerequisite_conditions": []}
   ```
9. **CRITICAL RULE** explained inline: `unjustified_knowledge` non-empty → `"valid": false`. Self-contradictory responses are explicitly warned against.

---

### 3.5 `prompts/swe_guided.j2`

Extends the standard `swe_gpt4.j2` 8-phase workflow. The key addition is the `<REFERENCE_INFORMATION>` block injected after the issue description, populated with `{{ instance.patch }}` and `{{ instance.test_patch }}`.

**REFERENCE_INFORMATION block (verbatim agent instructions)**:

```
RULES FOR USING THIS REFERENCE — READ CAREFULLY

ALLOWED — you may use the reference to:
- Know which module, file, or subsystem is likely relevant
- Understand the general nature of the bug
- Know which test cases are worth reproducing

FORBIDDEN — you must NEVER:
- Mention the reference patch or reference test explicitly
  (forbidden phrases: "looking at the reference patch", "the patch says",
   "according to the reference", "as specified in the patch")
- State any specific code value, variable name, exact line, or diff hunk
  without first having observed it through a tool call
- Skip exploration steps — all conclusions must be justified by tool call observations
- Claim to have "found" the bug/fix before actually reading the relevant source

HOW TO WORK CORRECTLY:
1. Use the reference silently to navigate to the relevant file/function
2. Read that file with a tool call — let the tool output be your evidence
3. Reason from what you observed in the tool output, not from the reference
4. Your stated fix must reference line numbers and values that appeared in YOUR tool output

If you violate these rules, your response will be rejected and you will be asked to redo it.
Repeated violations waste your iteration budget.
```

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
| `INSTRUCTION_TEMPLATE_NAME` | `swe_guided.j2` | Jinja2 template for task instruction |

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
| Agent repeatedly violated rules after rejection | Feedback message was too generic ("base your response on evidence") | Rewritten as 4-step numbered repair instructions |
| Agent leaked at step 0 on every instance | `<REFERENCE_INFORMATION>` block only said "guidance only, don't copy" | Block rewritten with explicit ALLOWED / FORBIDDEN / HOW TO WORK rules |

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
5. **Inspect results**:
   ```bash
   # Per-instance critic log
   cat {eval_output_dir}/blinded_critic_logs/<instance_id>.jsonl | python -m json.tool
   
   # Quick stats — how many rejections?
   jq -r '.valid' {eval_output_dir}/blinded_critic_logs/<instance_id>.jsonl | sort | uniq -c
   ```

---

## 8. Pending / Future Work

- Verify combined prompt fixes (REFERENCE_INFORMATION rules + feedback message repair instructions) result in fewer max-retry exhaustions in practice — a new evaluation run is needed.
- Consider adjusting `BLINDED_CRITIC_MAX_RETRIES` (currently 3) based on observed retry distributions.
- Evaluate whether a faster/cheaper model for the critic (e.g. `gpt-4o-mini`) gives sufficient precision, or if a stronger model is needed.
- Post-process `blinded_critic_logs/` across all instances to compute aggregate leakage statistics (rejection rate per step index, common unjustified knowledge types).
- Implement a notebook cell in `evaluation/evaluation_analysis/inspect_response.ipynb` that reads and visualises the per-instance critic JSONL files.
