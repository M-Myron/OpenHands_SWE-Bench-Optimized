# Oracle Guided Agent — Technical Session State

> This file captures the current state of the oracle_guided_codeact_agent module
> for session continuity. Read this first when resuming work.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  OracleGuidedCodeActAgent                │
│                  (oracle_guided_codeact_agent.py)        │
├─────────────────────────────────────────────────────────┤
│  step(state) → Action                                   │
│    1. Generate N solver candidates (blinded LLM)        │
│    2. Oracle Planner loop (with retries):               │
│       a. Planner decides: select / revise / rewrite     │
│       b. Phase gate check (action-based)                │
│       c. Critic validation (neural + symbolic)          │
│       d. If rejected → retry with feedback              │
│    3. Fact tracking + timing                            │
│    4. Convert to actions                                │
└─────────────────────────────────────────────────────────┘
         │                │                │
         ▼                ▼                ▼
  ┌────────────┐  ┌──────────────┐  ┌────────────┐
  │  Oracle     │  │   Hybrid     │  │   Fact     │
  │  Planner    │  │   Critic     │  │  Tracker   │
  │             │  │              │  │            │
  │ - plan()    │  │ - validate() │  │ - DAG mgmt │
  │ - sanitize  │  │ - realism    │  │ - phase    │
  │ - history   │  │   check      │  │   readiness│
  │ - save      │  │ - recheck    │  │ - ancestors│
  │   prompts   │  │ - symbolic   │  │ - node     │
  │             │  │   regex      │  │   resolve  │
  └────────────┘  └──────────────┘  └────────────┘
```

## File Inventory

| File | Purpose | Lines |
|------|---------|-------|
| `__init__.py` | Agent registration | ~5 |
| `oracle_guided_codeact_agent.py` | Main agent: step loop, phase gate, synthetic response | ~920 |
| `oracle_planner.py` | Planner LLM: plan(), parse, sanitize, decision history | ~400 |
| `hybrid_critic.py` | Critic: neural judgment, symbolic regex, realism check, recheck | ~420 |
| `fact_tracker.py` | Fact DAG: availability, usage, phase readiness, blocking ancestors | ~350 |
| `guided_config.py` | Config dataclasses + YAML loader | ~170 |
| `guided_config.default.yaml` | Reference config template | ~30 |
| `prompts/planner.j2` | Planner prompt: facts, history, candidates, rules, feedback | ~225 |
| `prompts/critic_judge.j2` | Critic judge: neural + symbolic extraction | ~100 |
| `prompts/critic_recheck.j2` | Critic recheck: re-evaluate failed regexes | ~70 |

## Key Design Decisions

### 1. Phase Gating (action-based, not keyword-based)
**Problem solved**: Keyword regex like `Phase 6|FIX IMPLEMENTATION` matches when solver merely *lists* phases in a plan.
**Solution**: Gate on actual tool actions:
- File creation (`str_replace_editor create`) → gates TEST_CREATION
- Code modification (`str_replace`, `insert`, `sed -i`) → gates FIX_IMPLEMENTATION
- Phase headers (`## Phase 5:`, `## Phase 6:`, `## Phase 7:`) → gates corresponding phase

### 1b. Oracle Auto-Activation
**Problem solved**: The oracle intervening from step 0 disrupts the solver's natural Phase 1-2 workflow (reading + running tests). The solver is forced into oracle-guided exploration before it can even restate the problem or check the test suite.
**Solution**: `oracle_auto_activate` mode delays oracle activation until the solver's response contains `## Phase 3` (exploration header), signaling it has naturally moved past reading/running. A fallback step threshold (`oracle_auto_activate_fallback_step`, default 5) ensures activation even if the solver never emits the header.
- Config: `agent.oracle_auto_activate: true`, `agent.oracle_auto_activate_fallback_step: 5`
- Env: `GUIDED_ORACLE_AUTO_ACTIVATE=1`, `GUIDED_ORACLE_AUTO_ACTIVATE_FALLBACK_STEP=5`
- Once activated, stays active for the rest of the session.

### 2. Fact ID Sanitization
**Problem solved**: Planner LLM echoes fact IDs (`f2 confirmed`, `[f5]`) into response_content which becomes SFT training data.
**Solution**: Post-processing regex in `_sanitize_response_content()` strips leaked IDs. Backup: critic's `_check_realism()` catches leaks in tool call args.

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
**Problem solved**: Phase gate feedback showed random available facts like `[f3]`, `[f4]`, `[f5]` instead of the specific facts blocking the artifact.
**Solution**: `get_blocking_ancestors()` walks DAG recursively to find leaf blockers — facts whose own dependencies are all met but that haven't been used yet.

### 7. Rejected Response in Feedback
**Problem solved**: Planner couldn't see what it just tried, so it repeated the same rejected action.
**Solution**: Append `## YOUR REJECTED RESPONSE (do NOT repeat this):` with truncated rejected text to feedback.

## Evaluation Pipeline

```bash
# Run inference on a single instance
bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_guided_infer.sh \
  llm.eval_glm5_fp8_t0 HEAD OracleGuidedCodeActAgent 1 100 1

# Key env vars
ORACLE_PREPROCESS_DIR=.../preprocess/test_v6  # v6 fact graphs
ORACLE_GUIDED_CONFIG=path/to/config.yaml      # optional
GUIDED_NUM_CANDIDATES=1
GUIDED_PLANNER_MAX_RETRIES=2
GUIDED_PLANNER_HISTORY_NEAR_WINDOW=5
GUIDED_ORACLE_AUTO_ACTIVATE=0                 # 1 = wait for Phase 3 header
GUIDED_ORACLE_AUTO_ACTIVATE_FALLBACK_STEP=5   # activate after this step if no Phase 3
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
