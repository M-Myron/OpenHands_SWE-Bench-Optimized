# Oracle Triad Agent — Usage Guide

**Module:** `openhands/agenthub/oracle_triad_codeact_agent/`  
**Evaluation Entry:** `evaluation/benchmarks/swe_bench_optimized/`

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Prerequisites](#2-prerequisites)
3. [Configuration](#3-configuration)
   - [config.toml Setup](#31-configtoml-setup)
   - [Environment Variables](#32-environment-variables)
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
9. [Advanced Configuration](#9-advanced-configuration)

---

## 1. Quick Start

```bash
# 1. Ensure config.toml has LLM sections (see §3.1)
# 2. Run evaluation on a single instance
bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
  openai/zai-org/GLM-5-FP8 \
  HEAD \
  OracleTriadCodeActAgent \
  1 \
  100 \
  1 \
  princeton-nlp/SWE-bench_Verified \
  test \
  1
```

This runs the Oracle Triad agent on the default test instance (`django__django-12663`) with:
- 3 debugger candidates per step
- 2 planner retries on validation failure
- The `verifier` validation backend (4.5-stage neuro-symbolic pipeline)
- Maximum 100 iterations

---

## 2. Prerequisites

- **Python 3.11+** with Poetry installed
- **Docker** (for SWE-bench runtime environments)
- **OpenHands** codebase at `/home/v-murongma/code/OpenHands_SWE-Bench-Optimized/`
- **LLM access** configured in `config.toml` (at minimum: primary model + oracle_planner + blinded_critic)
- **Preprocessing data** (optional but recommended): react facts and analysis markdown per instance

---

## 3. Configuration

### 3.1 config.toml Setup

The Oracle Triad requires **three LLM configurations** in `config.toml`:

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

**Notes:**
- All three can use the same model endpoint; the information barrier is enforced by prompt design, not model separation.
- The `oracle_planner` config is read by `ORACLE_PLANNER_LLM_CONFIG` env var (default: `oracle_planner`).
- The `blinded_critic` config is read by `ORACLE_PROPOSAL_CRITIC_LLM_CONFIG` env var (default: `blinded_critic`).

### 3.2 Environment Variables

All env vars have sensible defaults. Override only when needed.

#### Candidate Generation

| Variable | Default | Description |
|----------|---------|-------------|
| `BLINDED_DEBUGGER_NUM_CANDIDATES` | `3` | Number of debugger candidates per step. Higher values increase diversity but cost more LLM calls. Minimum: 1. |

#### Planner

| Variable | Default | Description |
|----------|---------|-------------|
| `ORACLE_PLANNER_LLM_CONFIG` | `oracle_planner` | LLM config section name in config.toml |
| `ORACLE_PLANNER_MAX_RETRIES` | `2` | How many revision attempts on validation failure. Minimum: 0. |
| `ORACLE_PLANNER_JSON_PARSE_MAX_RETRIES` | `3` | JSON response parsing retries |

#### Validation

| Variable | Default | Description |
|----------|---------|-------------|
| `PROPOSAL_VALIDATOR` | `verifier` | Validation backend. Options: `verifier` (4.5-stage), `critic` (one-shot), `none` |
| `USE_LEGACY_CRITIC` | `0` | Set to `1` to force `PROPOSAL_VALIDATOR=critic` |
| `ORACLE_PROPOSAL_CRITIC_LLM_CONFIG` | `blinded_critic` | LLM config for validator |
| `ORACLE_PROPOSAL_CRITIC_JSON_PARSE_MAX_RETRIES` | `3` | JSON parsing retries |

#### Verifier-Specific

| Variable | Default | Description |
|----------|---------|-------------|
| `VERIFIER_PROGRAMMATIC_ONLY` | `0` | Set to `1` to skip LLM-based claim extraction (use regex only) |
| `VERIFIER_EXTRACTOR_JSON_RETRIES` | `2` | LLM JSON parsing retries for claim extraction |
| `VERIFIER_LLM_CONFIG` | (uses critic config) | Override LLM config for verifier |

#### Data Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `ORACLE_PLANNER_CONTEXT_PATH` | (set per instance) | Path to oracle context JSON. Set automatically by evaluation runner. |
| `ORACLE_PREPROCESS_DIR` | Auto-detect | Directory with `{instance_id}_react_facts.json` and `{instance_id}_analysis.md`. Auto-detected from dataset/split. |

#### Debugging

| Variable | Default | Description |
|----------|---------|-------------|
| `ORACLE_PLANNER_SAVE_PROMPTS` | `0` | Set to `1` to save all planner prompts/responses |
| `ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS` | `0` | Set to `1` to save all validator prompts/responses |
| `ORACLE_PLANNER_SAVE_PROMPTS_DIR` | (set per instance) | Output directory for planner prompts |
| `ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS_DIR` | (set per instance) | Output directory for validator prompts |

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
| `DATASET` | 7 | `princeton-nlp/SWE-bench_Lite` | HuggingFace dataset identifier |
| `SPLIT` | 8 | `test` | Dataset split |
| `N_RUNS` | 9 | `1` | Runs per instance |

**Example — Full SWE-bench Verified evaluation:**

```bash
bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
  openai/zai-org/GLM-5-FP8 \
  HEAD \
  OracleTriadCodeActAgent \
  500 \
  100 \
  4 \
  princeton-nlp/SWE-bench_Verified \
  test \
  1
```

The shell script:
1. Auto-detects `ORACLE_PREPROCESS_DIR` from dataset/split
2. Sets default env vars for candidates, retries, validator mode
3. Hardcodes prompt saving (`ORACLE_PLANNER_SAVE_PROMPTS=1`)
4. Starts a background Docker cleanup loop (prunes every 30 minutes)
5. Calls the Python evaluation runner

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
# Set env vars for single-instance testing
export BLINDED_DEBUGGER_NUM_CANDIDATES=3
export ORACLE_PLANNER_MAX_RETRIES=2
export PROPOSAL_VALIDATOR=verifier
export ORACLE_PLANNER_SAVE_PROMPTS=1
export ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS=1
export INSTANCE_IDS=django__django-12663

bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
  openai/zai-org/GLM-5-FP8 HEAD OracleTriadCodeActAgent 1 100 1 \
  princeton-nlp/SWE-bench_Verified test 1
```

---

## 5. Oracle Context Preparation

### 5.1 Preprocessing Directory Structure

The evaluation runner auto-detects preprocessing data from:

```
evaluation/evaluation_outputs/outputs/<dataset_slug>-<split>/preprocess/
```

Where `<dataset_slug>` is the dataset path with `/` replaced by `__`.

**Example:**
```
evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/preprocess/
├── django__django-12663_react_facts.json
├── django__django-12663_analysis.md
├── astropy__astropy-14995_react_facts.json
├── astropy__astropy-14995_analysis.md
└── ...
```

### 5.2 React Facts Format

Each `{instance_id}_react_facts.json` file follows this schema:

```json
{
  "instance_id": "django__django-12663",
  "stages": [
    {
      "stage_name": "Initial Exploration",
      "steps": [
        {
          "goal": "Understand the issue context",
          "fact": "Read the django/db/models/lookups.py file to understand Lookup class",
          "preconditions": [],
          "reasoning": "The issue mentions Lookup subclass, we need to understand the base class",
          "action": "[TOOL CALL] read_file({\"path\": \"django/db/models/lookups.py\", \"view_range\": [1, 50]})",
          "observation": "Lookup class inherits from Expression mixin"
        },
        ...
      ]
    },
    {
      "stage_name": "Root Cause Analysis",
      "steps": [...]
    },
    ...
  ]
}
```

**Key fields:**
- `stages[].stage_name`: Investigation phase (e.g., "Initial Exploration", "Root Cause Analysis", "Fix Implementation")
- `steps[].preconditions`: List of history state requirements that must hold before this step is applicable. Used by the validator to check fact grounding.
- `steps[].action`: The concrete tool call. Used by the planner as a reference for proposal generation.
- `steps[].reasoning`: Why this step is appropriate. Aids planner in generating well-justified proposals.

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
  "referenced_fact_ids": ["stage1_2", "stage2_0"]
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
  "total_facts": 12,
  "used_facts": 4,
  "remaining_facts": 8,
  "used_fact_ids": ["stage1_0", "stage1_2", "stage2_0", "stage2_1"]
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

The shell launcher auto-detects this from the dataset/split. If running the Python entry point directly, set it manually:
```bash
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
  python -c "import sys,json; [print(f'step {json.loads(l)[\"step_index\"]}: {json.loads(l)[\"used_facts\"]}/{json.loads(l)[\"total_facts\"]}') for l in sys.stdin]"
```

### 8.3 Enabling Verbose Logging

Set the OpenHands log level to DEBUG for verbose output from all triad components:

```bash
export LOG_LEVEL=DEBUG
```

---

## 9. Advanced Configuration

### 9.1 Using Different Models Per Component

```toml
# Fast model for candidates
[llm]
model = "openai/gpt-4o-mini"

# Stronger model for oracle planning
[llm.oracle_planner]
model = "openai/gpt-4o"

# Cost-efficient model for validation
[llm.blinded_critic]
model = "openai/gpt-4o-mini"
```

### 9.2 Tuning Candidate Count

More candidates increase the chance of a good match but cost proportionally more LLM calls:

| `BLINDED_DEBUGGER_NUM_CANDIDATES` | Trade-off |
|:-:|-----------|
| 1 | Baseline — planner almost always proposes |
| 3 | Default — good diversity/cost balance |
| 5 | High diversity — useful for difficult instances |

### 9.3 Programmatic-Only Mode

For maximum speed (no LLM calls in verifier):

```bash
export VERIFIER_PROGRAMMATIC_ONLY=1
```

This uses regex-based claim extraction and skips LLM-assisted rule resolution and verdict synthesis. Only deterministic symbolic rules are evaluated. Faster but may produce more false positives (unnecessary rejections).

### 9.4 Disabling Validation for Ablation

```bash
export PROPOSAL_VALIDATOR=none
```

All planner proposals are accepted without verification. Useful for measuring the impact of the verifier on trajectory quality in controlled experiments.

### 9.5 Custom Instance Selection

```bash
# Single instance
export INSTANCE_IDS=django__django-12663

# Multiple instances (comma-separated)
export INSTANCE_IDS=django__django-12663,astropy__astropy-14995,sympy__sympy-20049
```

Or pass via command line:
```bash
poetry run python evaluation/benchmarks/swe_bench_optimized/run_infer_oracle_triad.py \
  ... \
  --instance-ids django__django-12663,astropy__astropy-14995
```
