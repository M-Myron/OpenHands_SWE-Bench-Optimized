# Oracle Triad Agent — Usage Guide

**Module:** `openhands/agenthub/oracle_triad_codeact_agent/`  
**Evaluation Entry:** `evaluation/benchmarks/swe_bench_optimized/`

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Prerequisites](#2-prerequisites)
3. [Configuration](#3-configuration)
   - [config.toml Setup (LLM endpoints)](#31-configtoml-setup-llm-endpoints)
   - [YAML Config File (triad settings)](#32-yaml-config-file-triad-settings)
   - [Environment Variable Overrides](#33-environment-variable-overrides)
   - [Configuration Priority](#34-configuration-priority)
   - [Stale Environment Variables](#35-stale-environment-variables)
4. [Running Evaluations](#4-running-evaluations)
   - [Shell Launcher](#41-shell-launcher)
   - [Python Entry Point](#42-python-entry-point)
   - [Single Instance Testing](#43-single-instance-testing)
5. [Oracle Context Preparation](#5-oracle-context-preparation)
   - [Preprocessing Directory Structure](#51-preprocessing-directory-structure)
   - [React Facts Format](#52-react-facts-format)
   - [Deep Analysis Format](#53-deep-analysis-format)
6. [Validator Selection](#6-validator-selection)
7. [Output Structure](#7-output-structure)
   - [Evaluation Output](#71-evaluation-output)
   - [Triad Log Format](#72-triad-log-format)
   - [Saved Prompts](#73-saved-prompts)
8. [Debugging and Troubleshooting](#8-debugging-and-troubleshooting)
9. [Experiment Recipes](#9-experiment-recipes)

---

## 1. Quick Start

```bash
# Minimal — uses all Python defaults (1 candidate, verifier, prompt saving on)
bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
  llm.eval_glm5_fp8_t0 HEAD OracleTriadCodeActAgent 1 100 1

# With a YAML config file for full control
ORACLE_TRIAD_CONFIG=my_experiment.yaml \
bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
  llm.eval_glm5_fp8_t0 HEAD OracleTriadCodeActAgent 1 100 1

# Override a single setting via env var (takes precedence over YAML)
BLINDED_DEBUGGER_NUM_CANDIDATES=3 \
bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
  llm.eval_glm5_fp8_t0 HEAD
```

---

## 2. Prerequisites

- **Python 3.11+** with Poetry installed
- **Docker** (for SWE-bench runtime environments)
- **OpenHands** codebase at `/home/v-murongma/code/OpenHands_SWE-Bench-Optimized/`
- **LLM access** configured in `config.toml` (at minimum: primary model + oracle_planner + blinded_critic)
- **Preprocessing data** (optional but recommended): react facts and analysis markdown per instance

---

## 3. Configuration

The Oracle Triad has three configuration layers, applied in priority order:

```
Explicit env var  >  YAML config file  >  Python defaults
```

### 3.1 config.toml Setup (LLM endpoints)

The Oracle Triad requires **three LLM configurations** in `config.toml`. These control which models and endpoints are used — they are separate from the YAML config that controls agent behaviour.

```toml
# Primary model — used by the Blinded Debugger
[llm]
model = "openai/zai-org/GLM-5-FP8"
api_key = "your-api-key"
base_url = "https://your-endpoint/v1"
temperature = 0.0

# Oracle Planner LLM — separate model with oracle access
[llm.oracle_planner]
model = "openai/zai-org/GLM-5-FP8"
api_key = "your-api-key"
base_url = "https://your-endpoint/v1"
temperature = 0.0

# Blinded Critic / Verifier LLM — no oracle access
[llm.blinded_critic]
model = "openai/zai-org/GLM-5-FP8"
api_key = "your-api-key"
base_url = "https://your-endpoint/v1"
temperature = 0.0
```

All three can use the same model endpoint; the information barrier is enforced by prompt design, not model separation.

### 3.2 YAML Config File (triad settings)

All agent behaviour, oracle context visibility, and prompt template sections can be controlled from a single YAML config file. **The YAML config is optional** — when not provided, all defaults come from Python dataclasses in `triad_config.py`, not from any file.

To use a YAML config:

```bash
export ORACLE_TRIAD_CONFIG=/path/to/my_config.yaml
```

A fully-documented reference template is at: `openhands/agenthub/oracle_triad_codeact_agent/triad_config.default.yaml`

This file is **never read automatically** — it's just a template to copy and edit. Only include the keys you want to change; missing keys fall back to the hardcoded Python defaults.

The config has four sections:

#### `oracle_context` — what the planner sees

Controls which private oracle information is assembled into the planner prompt.

| Key | Default | Description |
|-----|---------|-------------|
| `include_golden_patch` | `true` | Show the ground-truth code fix diff |
| `include_golden_test_patch` | `true` | Show the ground-truth test patch diff |
| `include_issue_understanding` | `true` | Show structured issue understanding (bug description, trigger, rationale) |
| `include_deep_analysis` | `true` | Show precomputed root-cause analysis markdown |
| `include_react_facts` | `true` | Load and display the investigation graph (stage3/stage2/legacy) |

#### `planner_prompt` — which template sections are rendered

Controls which instructional sections appear in the planner prompt template.

| Key | Default | Description |
|-----|---------|-------------|
| `include_tool_descriptions` | `true` | Show available tool catalog (names, params, descriptions) |
| `include_fact_usage_rules` | `true` | Show the "Complete investigation before implementation" rules |
| `include_finalize_guidance` | `true` | Show "Proceed to Finalize" guidance when all facts are consumed |
| `include_proposal_format` | `true` | Show proposal format instructions (REASONING + TOOL CALL) |
| `include_workflow_guidelines` | `true` | Show recommended 7-phase debugging workflow |

#### `agent` — runtime behaviour

| Key | Default | Description |
|-----|---------|-------------|
| `num_candidates` | `1` | Blinded debugger candidates per step |
| `planner_max_retries` | `2` | Planner revision retries on validation failure |
| `planner_history_window` | `5` | Recent action steps shown to planner (`-1` = full history) |
| `proposal_validator` | `verifier` | Validation backend: `verifier`, `critic`, or `none` |
| `planner_llm_config` | `oracle_planner` | config.toml section name for planner LLM |
| `critic_llm_config` | `blinded_critic` | config.toml section name for critic/verifier LLM |
| `verifier_llm_config` | `""` | config.toml section for verifier (empty = use critic) |
| `planner_json_parse_max_retries` | `3` | JSON parsing retries for planner response |
| `critic_json_parse_max_retries` | `3` | JSON parsing retries for critic/verifier response |
| `verifier_programmatic_only` | `false` | Skip LLM claim extraction, use regex only |
| `verifier_extractor_json_retries` | `2` | LLM JSON parsing retries for claim extraction |

#### `debug` — prompt saving

| Key | Default | Description |
|-----|---------|-------------|
| `save_planner_prompts` | `true` | Save all planner prompts/responses to disk |
| `save_critic_prompts` | `true` | Save all critic/verifier prompts/responses to disk |

#### Example: minimal config file

You only need to include the keys you want to change:

```yaml
# my_experiment.yaml — ablation without golden patches
oracle_context:
  include_golden_patch: false
  include_golden_test_patch: false

agent:
  num_candidates: 3
```

### 3.3 Environment Variable Overrides

Any setting from the YAML config can be overridden with an env var. This is useful for one-off changes without editing a file.

| Env Var | YAML Equivalent |
|---------|-----------------|
| `BLINDED_DEBUGGER_NUM_CANDIDATES` | `agent.num_candidates` |
| `ORACLE_PLANNER_MAX_RETRIES` | `agent.planner_max_retries` |
| `ORACLE_PLANNER_HISTORY_WINDOW` | `agent.planner_history_window` |
| `PROPOSAL_VALIDATOR` | `agent.proposal_validator` |
| `ORACLE_PLANNER_LLM_CONFIG` | `agent.planner_llm_config` |
| `ORACLE_PROPOSAL_CRITIC_LLM_CONFIG` | `agent.critic_llm_config` |
| `VERIFIER_LLM_CONFIG` | `agent.verifier_llm_config` |
| `VERIFIER_PROGRAMMATIC_ONLY` | `agent.verifier_programmatic_only` |
| `ORACLE_PLANNER_SAVE_PROMPTS` | `debug.save_planner_prompts` |
| `ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS` | `debug.save_critic_prompts` |

**Vars without YAML equivalents** (always set via env vars or shell args):

| Variable | Default | Description |
|----------|---------|-------------|
| `ORACLE_TRIAD_CONFIG` | (not set) | Path to YAML config file |
| `ORACLE_PREPROCESS_DIR` | Auto-detect | Directory with react facts and analysis files |
| `INSTANCE_IDS` | (all) | Comma-separated instance IDs to evaluate |

### 3.4 Configuration Priority

```
1. Explicit env var  (e.g., BLINDED_DEBUGGER_NUM_CANDIDATES=3)
       ↓ if not set
2. YAML config file  (e.g., agent.num_candidates: 3 in ORACLE_TRIAD_CONFIG)
       ↓ if not specified in YAML or no YAML file
3. Python defaults   (hardcoded in triad_config.py dataclasses, e.g., num_candidates=1)
```

The shell launcher only exports env vars the user explicitly set. If a var is unset, it is left for the Python-side `TriadConfig` to resolve from YAML or defaults.

**Where defaults live:**
- `ORACLE_PREPROCESS_DIR` — auto-detected by the shell script from dataset/split. Shows `(not set)` in the banner only if no preprocess directory exists on disk.
- `ORACLE_TRIAD_CONFIG` — genuinely optional. When unset, Python returns hardcoded defaults. `(not set)` in the banner is normal and expected.
- All other triad settings — hardcoded in Python dataclasses (`triad_config.py`). The `triad_config.default.yaml` file is a reference template, never auto-loaded.

### 3.5 Stale Environment Variables

Env vars from a previous shell session (e.g., `export BLINDED_DEBUGGER_NUM_CANDIDATES=3` run hours ago) persist silently and override both YAML config and Python defaults. The shell banner shows all active env var overrides so you can spot surprises:

```
  Triad env var overrides (unset = YAML/default):
    BLINDED_DEBUGGER_NUM_CANDIDATES=3    ← is this intentional or stale?
    PROPOSAL_VALIDATOR=none
```

To guarantee a clean slate, use `TRIAD_CLEAN_ENV=1`:

```bash
# Clear all triad env vars, use only YAML config + Python defaults
TRIAD_CLEAN_ENV=1 ORACLE_TRIAD_CONFIG=my.yaml \
  bash run_oracle_triad_infer.sh llm.eval_glm5_fp8_t0 HEAD

# Clear stale vars, then set one fresh override
TRIAD_CLEAN_ENV=1 BLINDED_DEBUGGER_NUM_CANDIDATES=5 \
  bash run_oracle_triad_infer.sh llm.eval_glm5_fp8_t0 HEAD
```

`TRIAD_CLEAN_ENV=1` unsets all triad-controlled env vars at the top of the script, before any other logic runs. Vars you set on the same command line (after `TRIAD_CLEAN_ENV=1`) are re-applied and take effect normally.

**Best practice:** Use YAML config for persistent settings. Use inline env vars (on the command line) for one-off overrides. Use `TRIAD_CLEAN_ENV=1` when unsure what's in your shell session.

---

## 4. Running Evaluations

### 4.1 Shell Launcher

```bash
bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
  <MODEL_CONFIG> <COMMIT_HASH> [AGENT] [EVAL_LIMIT] [MAX_ITER] \
  [NUM_WORKERS] [DATASET] [SPLIT] [N_RUNS]
```

| Argument | Position | Default | Description |
|----------|----------|---------|-------------|
| `MODEL_CONFIG` | 1 | (required) | LLM config name matching `[llm]` section |
| `COMMIT_HASH` | 2 | (required) | Git commit hash (use `HEAD` for current) |
| `AGENT` | 3 | `OracleTriadCodeActAgent` | Agent class name |
| `EVAL_LIMIT` | 4 | (all) | Number of instances to evaluate |
| `MAX_ITER` | 5 | `100` | Maximum iterations per instance |
| `NUM_WORKERS` | 6 | `1` | Parallel workers |
| `DATASET` | 7 | `SWE-Gym/SWE-Gym` | HuggingFace dataset identifier |
| `SPLIT` | 8 | `train` | Dataset split |
| `N_RUNS` | 9 | `1` | Runs per instance |

**Example — with YAML config:**

```bash
ORACLE_TRIAD_CONFIG=configs/no_patches.yaml \
bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
  llm.eval_glm5_fp8_t0 HEAD OracleTriadCodeActAgent 10 100 2
```

**Example — with env var overrides (no YAML):**

```bash
BLINDED_DEBUGGER_NUM_CANDIDATES=3 PROPOSAL_VALIDATOR=none \
bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
  llm.eval_glm5_fp8_t0 HEAD
```

The shell script:
1. Auto-detects `ORACLE_PREPROCESS_DIR` from dataset/split (tries `swegym_v5`, `swegym_v3`, then bare preprocess dir)
2. Only exports env vars the user explicitly set — unset vars are left for Python `TriadConfig` to resolve from YAML or defaults
3. Starts a background Docker cleanup loop (prunes every 30 minutes)
4. Calls the Python evaluation runner

### 4.2 Python Entry Point

For more control, call the Python runner directly:

```bash
poetry run python evaluation/benchmarks/swe_bench_optimized/run_infer_oracle_triad.py \
  --agent-cls OracleTriadCodeActAgent \
  --llm-config openai/zai-org/GLM-5-FP8 \
  --max-iterations 100 \
  --eval-num-workers 1 \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --mode swe \
  --n-runs 1 \
  --eval-n-limit 1 \
  --instance-ids django__django-12663
```

### 4.3 Single Instance Testing

For development and debugging:

```bash
# Option A: env vars
BLINDED_DEBUGGER_NUM_CANDIDATES=3 \
INSTANCE_IDS=django__django-12663 \
bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
  llm.eval_glm5_fp8_t0 HEAD OracleTriadCodeActAgent 1 100 1 \
  princeton-nlp/SWE-bench_Verified test+

# Option B: YAML config
ORACLE_TRIAD_CONFIG=configs/debug.yaml \
INSTANCE_IDS=django__django-12663 \
bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
  llm.eval_glm5_fp8_t0 HEAD
```

---

## 5. Oracle Context Preparation

### 5.1 Preprocessing Directory Structure

The evaluation runner auto-detects preprocessing data from:

```
evaluation/evaluation_outputs/outputs/<dataset_slug>-<split>/preprocess/
```

Where `<dataset_slug>` is the dataset path with `/` replaced by `__`.

Two directory layouts are supported:

**New graph-based layout (swegym_v5):**
```
evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/preprocess/swegym_v5/
├── bokeh__bokeh-12779/
│   ├── stage3_bridged.json          # Bridged investigation DAG (v5)
│   ├── stage2_facts.json            # Stage-2 graph (v3, fallback)
│   └── ...
└── ...
```

**Legacy flat layout:**
```
evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/preprocess/
├── django__django-12663_react_facts.json
├── django__django-12663_analysis.md
└── ...
```

The loader tries these paths in order:
1. `{preprocess_dir}/{instance_id}/stage3_bridged.json` (v5 bridged graph)
2. `{preprocess_dir}/{instance_id}/stage2_facts.json` (v3 graph)
3. `{preprocess_dir}/{instance_id}_react_facts.json` (legacy flat)

The shell launcher auto-detects the `swegym_v5` or `swegym_v3` subdirectory if present.

### 5.2 React Facts Format

Two formats are supported:

#### Bridged graph format (recommended — `stage3_bridged.json`)

A DAG of investigation nodes with typed categories, single evidence dict per node, motivation fields, and bridge facts:

```json
{
  "instance_id": "bokeh__bokeh-12779",
  "intention_groups": [...],
  "graph": [
    {
      "id": "f1",
      "category": "fact",
      "kind": "requirement_fact",
      "is_root": true,
      "grounding": "problem_rooted",
      "statement": "The problem statement reports a ResourceWarning...",
      "motivation": "The problem statement is the primary input for understanding what is broken.",
      "preconditions": [],
      "evidence": {
        "action": "[view] problem_statement",
        "observation": "The issue reports ResourceWarning at directory.py:126..."
      }
    },
    {
      "id": "f2",
      "category": "fact",
      "kind": "codebase_fact",
      "is_root": false,
      "grounding": "problem_rooted",
      "statement": "In directory.py line 126, open(init_py).read() creates an unclosed file...",
      "motivation": "Viewing the reported location confirms the unclosed file pattern.",
      "preconditions": ["f1"],
      "evidence": {
        "action": "[view] src/bokeh/application/handlers/directory.py 115-132",
        "observation": "Line 126: open() return value used inline without closing."
      }
    },
    {
      "id": "b1",
      "category": "bridge_fact",
      "discovery_type": "proactive_exploration",
      "kind": "codebase_fact",
      "statement": "Developer scans other example scripts and discovers scipy.misc.ascent()...",
      "preconditions": ["f10"],
      "evidence": {...}
    },
    {
      "id": "e1",
      "category": "edit_step",
      "title": "Fix unclosed file in DirectoryHandler.__init__",
      "intention_group": "g1",
      "file": "src/bokeh/application/handlers/directory.py",
      "preconditions": ["p1"],
      "evidence": {...}
    }
  ],
  "bridge_summary": {
    "total_non_problem_rooted": 6,
    "bridged": 3,
    "irreducible": 3,
    "bridges": [...],
    "irreducible_facts": [...]
  },
  "derivation_frontier": ["f1", "f2", ...]
}
```

**Node categories:** `fact`, `bridge_fact`, `organizational_fact`, `plan_fact`, `edit_step`, `validation_step`

**Key properties:**
- `preconditions`: List of **node IDs**. A node is only shown to the planner when all precondition nodes are fully consumed.
- `evidence`: **Single dict** `{action, observation}` (v5) — normalized internally to a 1-element list.
- `is_root`: True for facts with no preconditions (entry points to the DAG).
- `grounding`: `problem_rooted`, `bridged`, or `non_problem_rooted`.
- `motivation`: Why this fact was investigated.
- `discovery_type` (bridge_fact only): How the fact was discovered (`proactive_exploration`, `structural_browsing`).

#### Stage-2 graph format (`stage2_facts.json`)

Older DAG format with categories `trigger`/`base_fact` and evidence as an **array** of `{reasoning, action, observation}` dicts. Fully backward compatible — loaded and rendered identically.

#### Legacy format (`_react_facts.json`)

Flat stages with facts (backward compatible):

```json
{
  "stages": [
    {
      "stage": "phase_3_exploration",
      "goal": "Find the bug",
      "facts": [
        {
          "fact": "The bug is in lookups.py",
          "preconditions": ["Must have read the traceback"],
          "reasoning_action_observation": {
            "reasoning": "The traceback points to lookups.py",
            "action": "read_file path=lookups.py",
            "observation": "Found Lookup class"
          }
        }
      ]
    }
  ]
}
```

### 5.3 Deep Analysis Format

Each `{instance_id}_analysis.md` is a markdown document providing deeper understanding of the issue. It is loaded via `_load_preprocess_analysis()` and included in the oracle context as `deep_analysis`.

---

## 6. Validator Selection

Three validation backends are available:

### Verifier (Default: `PROPOSAL_VALIDATOR=verifier`)

The **History-Grounded Verifier** runs the full 4.5-stage neuro-symbolic pipeline:
1. **Claim extraction** — LLM decomposes proposal into typed claims
2. **Evidence retrieval** — Searches structured history memory
3. **Symbolic rules** — 14+ deterministic/LLM-assisted rules across 5 families
4. **LLM resolution** — Focused adjudication for ambiguous rule failures
5. **Verdict synthesis** — Final judgment with audit trail

**Best for:** Production evaluation, research experiments, SFT data quality assurance.

### Critic (Legacy: `PROPOSAL_VALIDATOR=critic`)

The **Oracle Proposal Critic** makes a single LLM call with the full history and proposal, asking for a JSON validity judgment. Simpler but less precise — no claim decomposition, no symbolic rule checking.

**Best for:** Quick development iteration, baseline comparisons.

### None (`PROPOSAL_VALIDATOR=none`)

Skips validation entirely. All planner proposals are materialized directly.

**Best for:** Testing planner behavior in isolation, ablation studies.

---

## 7. Output Structure

### 7.1 Evaluation Output

Output is written to:

```
evaluation/evaluation_outputs/outputs/<dataset_slug>-<split>/
  <agent_name>_<max_iter>_<model_name>_<eval_note>/
    ├── output.jsonl                              # Per-instance results
    ├── oracle_planner_context/
    │   └── <instance_id>.json                    # Oracle context per instance
    ├── oracle_planner_prompts/
    │   └── <instance_id>/
    │       ├── step_0005_attempt_00_plan.txt
    │       ├── step_0005_attempt_01_plan.txt
    │       └── ...
    ├── oracle_proposal_critic_prompts/
    │   └── <instance_id>/
    │       ├── step_0005_attempt_00_extraction.txt
    │       ├── step_0005_attempt_00_resolve_B4.txt
    │       ├── step_0005_attempt_00_synthesis.txt
    │       ├── step_0005_attempt_00_verdict.json
    │       └── ...
    └── oracle_triad_logs/
        └── <instance_id>.jsonl                   # Per-step triad log
```

### 7.2 Triad Log Format

Each line in `<instance_id>.jsonl` is a JSON object with one of these event types:

**Debugger candidate:**
```json
{
  "step_index": 5,
  "event": "debugger_candidate",
  "candidate_index": 0,
  "response_text": "Let me examine the Lookup class..."
}
```

**Planner decision:**
```json
{
  "step_index": 5,
  "event": "oracle_planner_decision",
  "attempt": 0,
  "decision": "proposal",
  "best_candidate_index": 1,
  "chosen_candidate_index": null,
  "proposal_response_text": "REASONING: The agent...\n[TOOL CALL] read_file(...)",
  "referenced_fact_ids": ["f1", "f2", "b1"]
}
```

**Verifier verdict (when `PROPOSAL_VALIDATOR=verifier`):**
```json
{
  "step_index": 5,
  "event": "verifier_verdict",
  "attempt": 0,
  "valid": false,
  "verdict": "invalid",
  "reason": "Rule B4 failed: line numbers not inferrable from context",
  "claims": [...],
  "rule_results": [...],
  "retrieved_unit_ids": [3, 7, 12],
  "suspected_leakage": [],
  "suggestion": "Read the file first to establish visible line context",
  "feedback_message": "[QA REVIEW - ORACLE PROPOSAL REJECTED]..."
}
```

**Critic validation (when `PROPOSAL_VALIDATOR=critic`):**
```json
{
  "step_index": 5,
  "event": "proposal_critic_validation",
  "attempt": 0,
  "valid": true,
  "reason": "All claims are grounded in history",
  "unjustified_knowledge": [],
  "prerequisite_conditions": []
}
```

**React fact usage summary:**
```json
{
  "step_index": 5,
  "event": "react_fact_usage_summary",
  "total_nodes": 45,
  "fully_used_nodes": 5,
  "partially_used_nodes": 1,
  "not_used_nodes": 39,
  "available_nodes": 9,
  "blocked_nodes": 31,
  "total_evidence": 53,
  "used_evidence": 6,
  "remaining_evidence": 47,
  "total_facts": 45,
  "used_facts": 5,
  "remaining_facts": 40,
  "used_fact_ids": ["t1", "t2", "t3", "t4", "t5"]
}
```

### 7.3 Saved Prompts

When prompt saving is enabled, prompts are saved as text files:

- **Planner prompts:** `step_{NNNN}_attempt_{NN}_plan.txt` — includes the rendered prompt and raw LLM response separated by a marker.
- **Verifier prompts:** Saved per stage:
  - `step_{NNNN}_attempt_{NN}_extraction.txt` — Claim extraction prompt + response
  - `step_{NNNN}_attempt_{NN}_resolve_{RULE_ID}.txt` — LLM resolution for specific rule
  - `step_{NNNN}_attempt_{NN}_synthesis.txt` — Verdict synthesis prompt + response
  - `step_{NNNN}_attempt_{NN}_verdict.json` — Full VerificationVerdict as JSON

---

## 8. Debugging and Troubleshooting

### 8.1 Common Issues

**Planner returns "candidate" every step (never proposes)**

The planner defaults to selecting candidates when it believes they are adequate. To encourage proposals:
- Check that oracle context is being loaded (look for `ORACLE_PLANNER_CONTEXT_PATH` in logs)
- Ensure react facts are present in the preprocessing directory
- Verify the planner prompt renders correctly (enable `ORACLE_PLANNER_SAVE_PROMPTS=1`)

**Verifier rejects all proposals**

Enable prompt saving and examine the verdict JSON:
```bash
cat oracle_proposal_critic_prompts/<instance_id>/step_0005_attempt_00_verdict.json | python -m json.tool
```

Check `rule_results` for which rules are failing. Common issues:
- A-family failures → agent hasn't followed phase ordering (e.g., proposing an edit before any analysis)
- B4 failures → line numbers in proposal lack context in history
- C2/C3 failures → planner is too specific about file+method localization

**React facts not loading**

Check that `ORACLE_PREPROCESS_DIR` is set:
```bash
echo $ORACLE_PREPROCESS_DIR
ls $ORACLE_PREPROCESS_DIR/<instance_id>_react_facts.json
```

The shell launcher auto-detects this from the dataset/split (including the `swegym_v3` subdirectory if present). If running the Python entry point directly, set it manually:
```bash
# For graph-based facts (swegym_v3 layout — instance subdirectories):
export ORACLE_PREPROCESS_DIR=evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/preprocess/swegym_v3
# For legacy flat layout:
export ORACLE_PREPROCESS_DIR=evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/preprocess
```

**JSON parse failures in planner/verifier**

These are retried automatically (default: 3 retries). If persistent:
- Check LLM response quality (save prompts to inspect raw output)
- Consider increasing `ORACLE_PLANNER_JSON_PARSE_MAX_RETRIES`
- Verify the LLM endpoint is responding correctly

### 8.2 Log Analysis

The triad log (`oracle_triad_logs/<instance_id>.jsonl`) provides a complete audit trail. Useful queries:

```bash
# Count proposals vs candidate selections
cat oracle_triad_logs/<id>.jsonl | grep oracle_planner_decision | \
  python -c "import sys,json; d=[json.loads(l) for l in sys.stdin]; \
  print(f'proposals: {sum(1 for x in d if x[\"decision\"]==\"proposal\")}'); \
  print(f'candidates: {sum(1 for x in d if x[\"decision\"]==\"candidate\")}')"

# Show all rejected proposals
cat oracle_triad_logs/<id>.jsonl | grep verifier_verdict | \
  python -c "import sys,json; [print(json.loads(l)['reason']) for l in sys.stdin if not json.loads(l)['valid']]"

# React fact consumption timeline
cat oracle_triad_logs/<id>.jsonl | grep react_fact_usage | \
  python -c "import sys,json; [print(f'step {json.loads(l)[\"step_index\"]}: avail={json.loads(l).get(\"available_nodes\",\"?\")}, used={json.loads(l)[\"used_facts\"]}/{json.loads(l)[\"total_facts\"]}, blocked={json.loads(l).get(\"blocked_nodes\",\"?\")}') for l in sys.stdin]"
```

### 8.3 Enabling Verbose Logging

Set the OpenHands log level to DEBUG for verbose output from all triad components:

```bash
export LOG_LEVEL=DEBUG
```

---

## 9. Experiment Recipes

All recipes below use a YAML config file. Save the YAML content to a file and run with `ORACLE_TRIAD_CONFIG=filename.yaml`.

### 9.1 Ablation: No Golden Patches

Test whether the planner can still guide effectively without seeing the ground-truth fix.

```yaml
# no_patches.yaml
oracle_context:
  include_golden_patch: false
  include_golden_test_patch: false
```

### 9.2 Ablation: Facts Only (No Patch, No Analysis)

Test the investigation graph in isolation.

```yaml
# facts_only.yaml
oracle_context:
  include_golden_patch: false
  include_golden_test_patch: false
  include_deep_analysis: false
```

### 9.3 Ablation: Patch Only (No React Facts)

Test planner with golden code fix but no investigation guidance.

```yaml
# patch_only.yaml
oracle_context:
  include_react_facts: false
  include_deep_analysis: false
```

### 9.4 Fast Mode (No Verifier, Minimal Overhead)

Maximum speed for initial development or stress testing.

```yaml
# fast.yaml
agent:
  num_candidates: 1
  planner_max_retries: 1
  proposal_validator: none
debug:
  save_planner_prompts: false
  save_critic_prompts: false
```

### 9.5 High-Quality Mode (More Candidates, Full Verification)

Maximum trajectory quality for final SFT data generation.

```yaml
# quality.yaml
agent:
  num_candidates: 3
  planner_max_retries: 3
  proposal_validator: verifier
```

### 9.6 Minimal Prompt (Strip Non-Essential Sections)

Reduce prompt token count for smaller models.

```yaml
# minimal_prompt.yaml
planner_prompt:
  include_tool_descriptions: false
  include_workflow_guidelines: false
  include_proposal_format: false
```

### 9.7 Using Different Models Per Component

Set different config.toml sections for each component:

```yaml
# multi_model.yaml
agent:
  planner_llm_config: oracle_planner_gpt4o  # strong model for planning
  critic_llm_config: critic_gpt4o_mini       # cost-efficient for validation
```

Requires corresponding `[llm.oracle_planner_gpt4o]` and `[llm.critic_gpt4o_mini]` sections in `config.toml`.

### 9.8 Programmatic-Only Verifier (No LLM in Verifier)

For maximum verifier speed — regex claim extraction + deterministic rules only.

```yaml
# regex_verifier.yaml
agent:
  verifier_programmatic_only: true
```

### 9.9 Combining YAML + Env Var Overrides

Use a base YAML config and override specific values per run:

```bash
# Base config: no patches, 1 candidate
# Override: use 3 candidates for this run only
BLINDED_DEBUGGER_NUM_CANDIDATES=3 \
ORACLE_TRIAD_CONFIG=no_patches.yaml \
bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
  llm.eval_glm5_fp8_t0 HEAD
```

The env var (`3`) wins over the YAML value (`1`), which wins over the Python default (`1`).
