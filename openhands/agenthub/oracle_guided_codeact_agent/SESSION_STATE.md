# Oracle Guided Agent — Technical Session State

> This file captures the current state of the oracle_guided_codeact_agent module
> for session continuity. Read this first when resuming work.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  OracleGuidedCodeActAgent  v2.0             │
│                  (oracle_guided_codeact_agent.py)           │
├─────────────────────────────────────────────────────────────┤
│  step(state) → Action                                      │
│    1. Auto-activation check (wait for Phase 3 or fallback) │
│    2. Generate N solver candidates (blinded LLM)            │
│    3. Oracle Planner loop (with retries + exp backoff):     │
│       a. Staged planner decides: select / revise / rewrite  │
│       b. Stage constraint check (stage-based gating)        │
│       c. Critic validation (neural + optional symbolic)     │
│       d. If rejected → retry with feedback                  │
│    4. Fact tracking (staged: explore→repro→plan→impl→fin)  │
│    5. Finish extension budget + force-finish logic          │
│    6. Convert to actions + triage logging                   │
└─────────────────────────────────────────────────────────────┘
         │                │                │
         ▼                ▼                ▼
  ┌────────────┐  ┌──────────────┐  ┌──────────────┐
  │  Oracle     │  │   Hybrid     │  │   Fact       │
  │  Planner    │  │   Critic     │  │  Tracker     │
  │             │  │              │  │              │
  │ - plan()    │  │ - validate() │  │ - 5-stage    │
  │ - staged    │  │ - realism    │  │   DAG mgmt   │
  │   prompts   │  │   check      │  │ - phase gate │
  │ - sanitize  │  │ - recheck    │  │ - impl       │
  │ - history   │  │ - symbolic   │  │   complete   │
  │ - save      │  │   (toggle)   │  │ - finish     │
  │   prompts   │  │ - fact       │  │   extension  │
  │ - JSON      │  │   relevance  │  │ - force      │
  │   parse     │  │ - transient  │  │   finish     │
  │   retry     │  │   retry      │  │ - skip facts │
  └────────────┘  └──────────────┘  └──────────────┘
```

## Stage System

The agent operates through 5 sequential stages, managed by FactTracker:

| Stage | Constant | Description |
|-------|----------|-------------|
| 1 | `STAGE_EXPLORATION` | Read code, understand the bug, explore the codebase |
| 2 | `STAGE_REPRODUCTION` | Write/run reproduction scripts to confirm the bug |
| 3 | `STAGE_ANALYSIS_PLANNING` | Analyze root cause, plan the fix |
| 4 | `STAGE_IMPLEMENTATION_VERIFICATION` | Apply code edits, run validation |
| 5 | `STAGE_FINISH` | Submit the fix |

The staged planner template (`planner_staged.j2`) provides stage-specific instructions, fact rendering, and artifact marking rules.

## File Inventory

| File | Purpose | Lines |
|------|---------|-------|
| `__init__.py` | Agent registration | ~6 |
| `oracle_guided_codeact_agent.py` | Main agent: step loop, stage gate, synthetic response, triage | ~1400 |
| `oracle_planner.py` | Planner LLM: plan(), staged prompts, parse, sanitize, decision history | ~450 |
| `hybrid_critic.py` | Critic: neural judgment, symbolic regex (toggleable), realism, recheck, fact relevance, transient retry | ~720 |
| `fact_tracker.py` | Fact DAG: 5-stage lifecycle, availability, usage, impl-complete, finish extension, force-finish, skip-facts | ~1370 |
| `guided_config.py` | Config dataclasses + YAML loader + env export | ~196 |
| `guided_config.yaml` | Active config (overrides defaults) | ~31 |
| `guided_config.default.yaml` | Reference config template (all defaults) | ~37 |
| `prompts/planner_staged.j2` | **Primary** staged planner prompt: stage-specific facts, rules, artifact marking | ~418 |
| `prompts/planner.j2` | Legacy planner prompt (non-staged) | ~222 |
| `prompts/planner_retry.j2` | Retry-specific planner prompt (empty/unused) | 0 |
| `prompts/planner_stage1_exploration.j2` | Stage-1 exploration prompt (empty/unused) | 0 |
| `prompts/critic_judge.j2` | Critic judge: neural + conditional symbolic extraction | ~107 |
| `prompts/critic_recheck.j2` | Critic recheck: re-evaluate failed regexes | ~75 |

## Config System

### Dataclass Fields (`guided_config.py`)

| Section | Field | Type | Default | Env Var |
|---------|-------|------|---------|---------|
| **OracleContextConfig** | `include_golden_patch` | bool | True | `GUIDED_INCLUDE_GOLDEN_PATCH` |
| | `include_golden_test_patch` | bool | True | `GUIDED_INCLUDE_GOLDEN_TEST_PATCH` |
| | `include_issue_understanding` | bool | True | `GUIDED_INCLUDE_ISSUE_UNDERSTANDING` |
| | `include_react_facts` | bool | True | `GUIDED_INCLUDE_REACT_FACTS` |
| **PlannerConfig** | `history_near_window` | int | 5 | `GUIDED_PLANNER_HISTORY_NEAR_WINDOW` |
| | `include_system_instruction` | bool | True | `GUIDED_PLANNER_INCLUDE_SYSTEM_INSTRUCTION` |
| | `llm_config` | str | `'oracle_planner'` | — |
| | `json_parse_max_retries` | int | 3 | `GUIDED_PLANNER_JSON_PARSE_MAX_RETRIES` |
| **CriticConfig** | `llm_config` | str | `'blinded_critic'` | — |
| | `json_parse_max_retries` | int | 3 | `GUIDED_CRITIC_JSON_PARSE_MAX_RETRIES` |
| | `enable_symbolic_checks` | bool | True | `GUIDED_CRITIC_ENABLE_SYMBOLIC_CHECKS` |
| **AgentConfig** | `num_candidates` | int | 1 | `GUIDED_NUM_CANDIDATES` |
| | `planner_max_retries` | int | 2 | `GUIDED_PLANNER_MAX_RETRIES` |
| | `gate_max_retries` | int | 2 | `GUIDED_GATE_MAX_RETRIES` |
| | `oracle_start_step` | int | 0 | `GUIDED_ORACLE_START_STEP` |
| | `oracle_auto_activate` | bool | False | `GUIDED_ORACLE_AUTO_ACTIVATE` |
| | `oracle_auto_activate_fallback_step` | int | 5 | `GUIDED_ORACLE_AUTO_ACTIVATE_FALLBACK_STEP` |
| | `finish_extension_steps` | int | 10 | `GUIDED_FINISH_EXTENSION_STEPS` |
| | `transient_retries` | int | 5 | `GUIDED_TRANSIENT_RETRIES` |
| | `retry_base_wait` | int | 10 | `GUIDED_RETRY_BASE_WAIT` |
| **DebugConfig** | `save_planner_prompts` | bool | True | — |
| | `save_critic_prompts` | bool | True | — |

### Active Config Overrides (`guided_config.yaml`)

```yaml
oracle_context:
  include_golden_patch: false
  include_golden_test_patch: false
  include_issue_understanding: false
planner:
  include_system_instruction: false
critic:
  json_parse_max_retries: 9
agent:
  oracle_auto_activate: true
  oracle_auto_activate_fallback_step: 10
  transient_retries: 15
```

## Key Design Decisions

### 1. Stage-Based Gating (replaces simple phase gating)
**Problem solved**: Keyword regex like `Phase 6|FIX IMPLEMENTATION` matches when solver merely *lists* phases in a plan.
**Solution**: Stage constraint system (`_GATE_STAGE_CONSTRAINT`) gates on actual tool actions and stage progression:
- File creation (`str_replace_editor create`) → gates TEST_CREATION
- Code modification (`str_replace`, `insert`, `sed -i`) → gates FIX_IMPLEMENTATION
- Phase headers (`## Phase 5:`, `## Phase 6:`, `## Phase 7:`) → gates corresponding phase
- Stage transitions tracked by FactTracker's `get_current_stage()` method

### 1b. Oracle Auto-Activation
**Problem solved**: The oracle intervening from step 0 disrupts the solver's natural Phase 1-2 workflow.
**Solution**: `oracle_auto_activate` mode delays oracle activation until the solver's response contains `## Phase 3` (exploration header), signaling it has naturally moved past reading/running. A fallback step threshold ensures activation even if the solver never emits the header.
- Config: `agent.oracle_auto_activate: true`, `agent.oracle_auto_activate_fallback_step: 5`
- Env: `GUIDED_ORACLE_AUTO_ACTIVATE=1`, `GUIDED_ORACLE_AUTO_ACTIVATE_FALLBACK_STEP=5`
- Once activated, stays active for the rest of the session.

### 2. Fact ID Sanitization
**Problem solved**: Planner LLM echoes fact IDs (`f2 confirmed`, `[f5]`) into response_content which becomes SFT training data.
**Solution**: Post-processing regex in `_sanitize_response_content()` strips leaked IDs. Backup: critic's `_check_realism()` catches leaks in tool call args (4 compiled regex patterns in `_REALISM_PATTERNS`).

### 3. Node ID Resolution
**Problem solved**: Planner outputs `reproduce_script` instead of `repro1`, `issue_analysis` instead of `analysis`.
**Solution**: `_resolve_node_id()` in FactTracker matches by `node_type` as fallback. Partial prefix matching for compound names like `code_edit_exception`.

### 4. Decision History (accepted only)
**Problem solved**: Recording all decisions (including rejected retries) polluted the planning continuity.
**Solution**: `record_accepted_decision()` called only at 3 acceptance points (select, critic pass, no-critic accept). Rejected attempts excluded.

### 5. Recheck Only on Disagreement
**Problem solved**: Running recheck LLM when neural already says invalid wastes tokens.
**Solution**: Recheck only when `neural_valid=True AND symbolic_failures > 0` (the disagreement case needing tiebreaker).

### 6. Blocking Ancestors (not random available facts)
**Problem solved**: Phase gate feedback showed random available facts instead of the specific facts blocking the artifact.
**Solution**: `get_blocking_ancestors()` walks DAG recursively to find leaf blockers — facts whose own dependencies are all met but that haven't been used yet.

### 7. Rejected Response in Feedback
**Problem solved**: Planner couldn't see what it just tried, so it repeated the same rejected action.
**Solution**: Append `## YOUR REJECTED RESPONSE (do NOT repeat this):` with truncated rejected text to feedback.

### 8. Transient Retry with Exponential Backoff
**Problem solved**: LLM 502/APIError crashes killed entire runs. Hardcoded 3×10s retries were insufficient.
**Solution**: Configurable exponential backoff (default 5 retries: 10→20→40→80→160s) applied to:
- `oracle_guided_codeact_agent.py`: candidate generation
- `oracle_planner.py`: planner LLM calls
- `hybrid_critic.py`: critic LLM calls via `_llm_call_with_transient_retry()`
- Also added `litellm.APIError` to `LLM_RETRY_EXCEPTIONS` in `openhands/llm/llm.py`

### 9. Conditional Symbolic Checks
**Problem solved**: Symbolic regex checks in critic produced false positives and were sometimes counterproductive.
**Solution**: `enable_symbolic_checks` toggle (config + env var `GUIDED_CRITIC_ENABLE_SYMBOLIC_CHECKS`). When disabled, critic_judge.j2 template conditionally omits Part 2 symbolic checks section and output format.

### 10. Reasoning Text Quality Rules
**Problem solved**: SFT training data contained repetitive, verbose reasoning that trained bad habits.
**Solution**: Added "Reasoning text quality" subsections to both `planner_staged.j2` (in exploration and stages 2-4 sections) and `planner.j2`. Rules: no repetition, concise, logically connected, no redundancy, natural voice.

### 11. Artifact Marking Rules (Implementation Stage)
**Problem solved**: Planner failed to properly mark code_edit and validation artifacts during implementation.
**Solution**: Added detailed `### Artifact marking rules (REQUIRED)` section in `planner_staged.j2` for the `implementation_verification` stage with rules for code_edit artifacts, validation artifacts, and multi-marking.

### 12. Finish Extension Budget
**Problem solved**: Agent would hit max_iterations before completing validation after implementation.
**Solution**: `finish_extension_steps` (default 10) gives extra steps after implementation is detected as complete. `check_and_set_impl_complete()` and `set_finish_extension_budget()` manage the extension. `should_force_finish()` triggers when budget is exhausted.

### 13. Fact Relevance Checking
**Problem solved**: Critic needed to verify that the planner's response actually used relevant facts.
**Solution**: `check_facts_relevance()` method in HybridCritic validates fact usage in planner output.

## FactTracker Key Methods

The FactTracker (1,374 lines) manages the 5-stage fact DAG lifecycle:

| Method | Purpose |
|--------|---------|
| `get_current_stage()` | Determine which stage based on fact usage state |
| `render_categorized_facts()` | Group facts by category for the current stage |
| `render_implementation_stage_nodes()` | Render code_edit and repro nodes for impl stage |
| `check_fact_unlocker_satisfied()` | Check if a fact's unlocker condition is met in history |
| `categorize_available_facts()` | Sort facts into used/available/blocked |
| `render_solving_summary()` | Generate a summary of the solving progress |
| `should_force_finish()` | Check if the finish extension budget is exhausted |
| `check_and_set_impl_complete()` | Detect when implementation is done |
| `set_finish_extension_budget()` | Set extra step budget post-implementation |
| `get_unexplored_fact_breakdown()` | Detailed breakdown of unexplored facts |
| `get_unexplored_fact_summary()` | Summary text for unexplored facts |
| `skip_remaining_facts()` | Mark remaining facts as skipped |
| `get_unused_fact_statements()` | Get statements of unused facts |
| `get_blocking_ancestors()` | Find leaf blocker facts in DAG |

## Evaluation Pipeline

```bash
# Run inference on a single instance
bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_guided_infer.sh \
  llm.eval_glm5_fp8_t0 HEAD OracleGuidedCodeActAgent 1 100 1

# Instance-major scheduling (batch inference)
bash evaluation/benchmarks/swe_bench_optimized/scripts/swegym/run_oracle_guided_infer_instance_major.sh

# Key env vars
ORACLE_PREPROCESS_DIR=.../preprocess/test_v6  # v6 fact graphs
ORACLE_GUIDED_CONFIG=path/to/config.yaml      # optional
GUIDED_NUM_CANDIDATES=1
GUIDED_PLANNER_MAX_RETRIES=2
GUIDED_PLANNER_HISTORY_NEAR_WINDOW=5
GUIDED_ORACLE_AUTO_ACTIVATE=0                 # 1 = wait for Phase 3 header
GUIDED_ORACLE_AUTO_ACTIVATE_FALLBACK_STEP=5   # activate after this step if no Phase 3
GUIDED_TRANSIENT_RETRIES=5                    # LLM transient retry count
GUIDED_RETRY_BASE_WAIT=10                     # base wait in seconds (exponential)
GUIDED_CRITIC_ENABLE_SYMBOLIC_CHECKS=1        # 0 to disable symbolic regex checks
GUIDED_FINISH_EXTENSION_STEPS=10              # extra steps after impl complete
```

## Monitoring

```bash
# Live inference monitor (Python-powered bash script)
./monitor_infer.sh <logs_dir> [refresh_sec] [threshold] [num_lines]
# num_lines=0: compact table mode, 1+: detail mode with log tails
```

## Fact Graph Format (v6)

```json
{
  "instance_id": "getmoto__moto-4787",
  "nodes": [
    {
      "id": "f1",
      "node_type": "fact",
      "type": "static",
      "statement": "The problem statement reports...",
      "unlocker": {"action": "[view] problem_statement", "observation": "..."},
      "depends_on": []
    },
    {
      "id": "repro1",
      "node_type": "reproduce_script",
      "description": "Demonstrate the bug...",
      "code": "import boto3\n...",
      "output_before_fix": "BUG: ...",
      "output_after_fix": "PASS: ...",
      "depends_on": ["f1", "f15"]
    },
    {
      "id": "edit1",
      "node_type": "code_edit",
      "file": "moto/dynamodb2/exceptions.py",
      "old_str": "class EmptyKeyAttributeException...",
      "new_str": "class MultipleTransactionsException...\n\nclass EmptyKeyAttributeException...",
      "depends_on": ["plan"]
    }
  ]
}
```

## Known Issues & Next Steps

### Critical
1. **Long runs on complex graphs** — 44-node fact graph (moto-4881) caused 84 steps, 5+ hours. Need a mechanism to skip/collapse non-essential facts when the graph is too deep.
2. **Node ID resolution ambiguity** — `_resolve_node_id("code_edit")` returns first `code_edit` node. When there are 9 edit nodes, the planner can't target specific ones by type name.

### Important
3. **Planner repeatedly tries to skip phases** — Even with clear feedback, GLM-5 sometimes proposes implementation before completing analysis/plan. The mandatory correction section helps but isn't 100%.
4. **Sanitizer false positives** — Pattern `f\d{1,2}` can match legitimate text like "f1 or f2 arguments" or Python f-strings. Need negative lookbehind for common false-positive contexts.
5. **Phase gate doesn't cover `think` with implementation content** — Solver can use `think` tool to write a complete fix plan that includes code, bypassing the phase gate.

### Nice to Have
6. Add per-instance timeout to prevent 5-hour runs
7. Track token usage per component (planner vs critic vs solver)
8. Add a "fast mode" that skips reproduction and goes straight to implementation for simple fixes
9. Support multiple candidates from different models for diversity
