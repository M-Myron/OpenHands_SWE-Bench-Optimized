# Oracle Triad CodeAct Agent

A three-component architecture for oracle-guided SWE-bench evaluation that separates concerns between a **blinded debugger**, an **oracle-aware planner**, and a **proposal validator** — ensuring the final agent output is history-grounded, non-leaky, and follows a strict debugging workflow.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Components](#components)
  - [OracleTriadCodeActAgent](#oracletriadcodeactagent)
  - [OraclePlanner](#oracleplanner)
  - [Proposal Validators](#proposal-validators)
    - [HistoryGroundedVerifier (default)](#historygroundedverifier-default)
    - [OracleProposalCritic (legacy)](#oracleproposalcritic-legacy)
  - [Supporting Modules](#supporting-modules)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [config.toml LLM Sections](#configtoml-llm-sections)
  - [Selecting a Validator](#selecting-a-validator)
- [Running Evaluations](#running-evaluations)
  - [Quick Start](#quick-start)
  - [Shell Launcher](#shell-launcher)
  - [Python Entry Point](#python-entry-point)
  - [Oracle Context Preparation](#oracle-context-preparation)
- [Step-by-Step Execution Flow](#step-by-step-execution-flow)
- [Prompt Templates](#prompt-templates)
- [Triad Log Format](#triad-log-format)
- [File Index](#file-index)

---

## Overview

The Oracle Triad agent addresses a fundamental tension in oracle-guided SWE-bench evaluation: the oracle has access to the ground-truth patch and test, but the generated debugging trajectory must appear as if the agent discovered the fix independently. Leaking oracle knowledge into the trajectory produces unrealistic training data.

The triad architecture solves this by splitting the work:

| Role | LLM Access | Oracle Access | Purpose |
|------|-----------|---------------|---------|
| **Blinded Debugger** | Primary model | None | Generates realistic candidate debugging actions |
| **Oracle Planner** | Separate model | Full (patch, tests, analysis) | Steers toward the correct fix without leaking |
| **Proposal Validator** | Separate model | None (history only) | Rejects proposals that leak oracle knowledge |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                   OracleTriadCodeActAgent.step()                      │
│                                                                       │
│  1. Blinded Debugger generates N candidates (no oracle)               │
│                                                                       │
│  2. Oracle Planner selects best candidate OR proposes revised action  │
│     ├─ Has full oracle context (patch, test, analysis, react facts)   │
│     └─ Enforces phase ordering and SFT-quality constraints            │
│                                                                       │
│  3. Proposal Validator checks for leakage/grounding                   │
│     ├─ PROPOSAL_VALIDATOR=verifier  →  4-stage neuro-symbolic         │
│     │   ┌──────────────────────────────────────────────────┐          │
│     │   │  Stage 1: Extract claims & preconditions         │          │
│     │   │  Stage 2: Retrieve evidence from history         │          │
│     │   │  Stage 3: Apply symbolic rules (A–E families)    │          │
│     │   │  Stage 4: Synthesize verdict                     │          │
│     │   └──────────────────────────────────────────────────┘          │
│     ├─ PROPOSAL_VALIDATOR=critic    →  One-shot LLM critic            │
│     └─ PROPOSAL_VALIDATOR=none      →  Skip validation                │
│                                                                       │
│  4. On rejection → feedback to planner → retry (up to max retries)    │
│  5. On exhaustion → fallback to planner's best candidate              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### OracleTriadCodeActAgent

**File:** `oracle_triad_codeact_agent.py`

The main agent class, extending `CodeActAgent`. Orchestrates the blinded debugger → oracle planner → validator loop on each `step()` call.

**Key behavior:**
- Generates `N` candidate responses from the blinded debugger (primary LLM, no oracle access)
- Passes candidates + full interaction history to the oracle planner
- If planner proposes a revised response, validates it through the selected validator
- On rejection, feeds structured feedback back to the planner for retry
- Falls back to the planner's best candidate if retries are exhausted
- Accepted proposals are materialized by re-calling the debugger LLM with injected guidance

### OraclePlanner

**File:** `oracle_planner.py`

The oracle-aware component. Has full access to the ground-truth patch, test patch, issue understanding, deep analysis, and react facts. Its job is to steer the debugging trajectory toward the correct fix without leaking oracle-specific details.

**Key classes:**
- `OraclePlanner` — renders `planner_select_or_propose.j2`, returns a `PlannerDecision`
- `PlannerDecision` — dataclass with `decision` (`'candidate'` or `'proposal'`), selected index, proposal text, referenced fact IDs
- `ReactFactTracker` — manages structured investigation facts from preprocessing; tracks which facts have been used to avoid repetition

### Proposal Validators

#### HistoryGroundedVerifier (default)

**File:** `verifier.py`

A 4-stage neuro-symbolic pipeline that validates proposals using only the interaction history — no filesystem access, no shell commands:

| Stage | Method | Description |
|-------|--------|-------------|
| 1. Extract | `ClaimExtractor` or `ProgrammaticClaimExtractor` | Decompose proposal into structured claims, preconditions, and a retrieval plan |
| 2. Retrieve | `StructuredHistoryMemory` search methods | Fetch evidence units from indexed history using `file:`, `keyword:`, `phase:`, `tag:`, `symbol:` queries |
| 3. Rules | `SymbolicRuleEngine.evaluate_all()` | Apply 14 deterministic rules across 5 families (A–E) |
| 4. Synthesize | Deterministic or LLM-assisted | High-severity failure → reject immediately. All pass → accept. Ambiguous leakage → LLM synthesis |

Returns a `VerificationVerdict` with `.valid` property for backward compatibility.

**Symbolic Rule Families:**

| Family | Focus | Severity | Rules |
|--------|-------|----------|-------|
| **A** — Workflow | Phase ordering enforcement | High | A1: edit needs analysis, A2: verify needs implementation, A3: finalize needs verification, A4: phase evidence |
| **B** — Reachability | File/symbol discoverability | High/Medium | B1: file path justification, B2: symbol justification, B3: edit target read, B4: symbol definition |
| **C** — Leakage | Oracle knowledge detection | High | C1: hidden implementation detail, C2: unsupported localization, C3: oracle-only dependence |
| **D** — Evidence | Claim support adequacy | Medium | D1: bug cause support, D2: fix claim support |
| **E** — Discoverability | Advisory suggestions | Low | E1: discoverable next step, E2: missing prerequisite |

#### OracleProposalCritic (legacy)

**File:** `proposal_critic.py`

A one-shot LLM-based validator. Renders `validate_oracle_proposal.j2` and asks the LLM to judge whether the proposal is grounded, non-leaky, and follows the workflow. Returns `ProposalValidationResult`.

Simpler but less interpretable — the LLM produces a `valid: true/false` judgment without structured decomposition.

### Supporting Modules

| File | Purpose |
|------|---------|
| `history_memory.py` | `StructuredHistoryMemory` — indexes the OpenHands event stream into searchable `HistoryUnit` objects with file/symbol extraction, phase detection, and tag classification |
| `claim_extractor.py` | `ClaimExtractor` (LLM-assisted) and `ProgrammaticClaimExtractor` (regex fallback) — extract structured claims, preconditions, proof obligations, and retrieval plans from proposal text |
| `symbolic_rules.py` | `SymbolicRuleEngine` — evaluates 14 deterministic rules across 5 families, returns `RuleResult` list with severity, related claims, and evidence |

---

## Configuration

### Environment Variables

#### Agent Core

| Variable | Default | Description |
|----------|---------|-------------|
| `BLINDED_DEBUGGER_NUM_CANDIDATES` | `3` | Number of candidate responses per step |
| `ORACLE_PLANNER_MAX_RETRIES` | `2` | Max planner retry attempts on proposal rejection |
| `PROPOSAL_VALIDATOR` | `verifier` | Validator backend: `verifier`, `critic`, or `none` |
| `USE_LEGACY_CRITIC` | `0` | Legacy compat — set to `1` as shorthand for `PROPOSAL_VALIDATOR=critic` |

#### LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ORACLE_PLANNER_LLM_CONFIG` | `oracle_planner` | config.toml section for the planner LLM |
| `ORACLE_PROPOSAL_CRITIC_LLM_CONFIG` | `blinded_critic` | config.toml section for the legacy critic LLM |
| `VERIFIER_LLM_CONFIG` | *(falls back to critic config)* | config.toml section for the verifier LLM |
| `CONFIG_FILE` | `config.toml` | Path to the LLM configuration file |

#### Verifier-Specific

| Variable | Default | Description |
|----------|---------|-------------|
| `VERIFIER_PROGRAMMATIC_ONLY` | `0` | Force programmatic-only claim extraction (skip LLM) |
| `VERIFIER_EXTRACTOR_JSON_RETRIES` | `2` | Max JSON parse retries for LLM claim extraction |

#### Oracle Context

| Variable | Default | Description |
|----------|---------|-------------|
| `ORACLE_PLANNER_CONTEXT_PATH` | *(empty)* | Path to JSON with oracle context (patch, tests, analysis, react facts) |
| `ORACLE_PREPROCESS_DIR` | *(empty)* | Directory with `{instance_id}_analysis.md` and `{instance_id}_react_facts.json` |

#### Prompt Saving

| Variable | Default | Description |
|----------|---------|-------------|
| `ORACLE_PLANNER_SAVE_PROMPTS` | `0` | Enable planner prompt saving |
| `ORACLE_PLANNER_SAVE_PROMPTS_DIR` | *(empty)* | Directory for saved planner prompts |
| `ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS` | `0` | Enable critic/verifier prompt saving |
| `ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS_DIR` | *(empty)* | Directory for saved critic/verifier prompts |

#### JSON Parse Retries

| Variable | Default | Description |
|----------|---------|-------------|
| `ORACLE_PLANNER_JSON_PARSE_MAX_RETRIES` | `3` | Max retries for planner JSON parsing |
| `ORACLE_PROPOSAL_CRITIC_JSON_PARSE_MAX_RETRIES` | `3` | Max retries for critic JSON parsing |

### config.toml LLM Sections

```toml
# Blinded critic / verifier LLM (no oracle access)
[llm.blinded_critic]
model = "openai/your-model-name"
api_key = "your-api-key"
base_url = "http://localhost:8000/v1"
temperature = 0
max_output_tokens = 8192
timeout = 600

# Oracle planner LLM (has oracle access via prompt)
[llm.oracle_planner]
model = "openai/your-model-name"
api_key = "your-api-key"
base_url = "http://localhost:8000/v1"
temperature = 0
max_output_tokens = 8192
timeout = 600
```

Both sections can use different models. The planner and critic/verifier are intentionally separate LLMs so that the validator has no oracle context in its weights or prompt.

### Selecting a Validator

```bash
# Use the neuro-symbolic verifier (default)
export PROPOSAL_VALIDATOR=verifier

# Use the legacy one-shot LLM critic
export PROPOSAL_VALIDATOR=critic

# Disable proposal validation entirely
export PROPOSAL_VALIDATOR=none
```

When `PROPOSAL_VALIDATOR` is set, it takes precedence. If unset, `USE_LEGACY_CRITIC=1` can be used as a fallback to select the legacy critic.

**Comparison:**

| Aspect | `verifier` | `critic` |
|--------|-----------|---------|
| Method | 4-stage pipeline with symbolic rules | Single LLM prompt |
| Interpretability | Structured: rule IDs, claim IDs, evidence trails | Free-text reason |
| LLM calls | 0–2 per validation (extraction + optional synthesis) | 1 per validation |
| Deterministic rules | 14 rules across 5 families | None (LLM-only) |
| Feedback quality | Structured with specific rule violations and suggestions | Free-text |
| Oracle isolation | No filesystem, no shell, history-only retrieval | Prompt-based |

---

## Running Evaluations

### Quick Start

```bash
# 1. Ensure config.toml has [llm.oracle_planner] and [llm.blinded_critic] sections

# 2. Run with default settings (verifier, 3 candidates, 2 retries)
bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
  oracle_planner \
  $(git rev-parse HEAD)

# 3. Run with legacy critic
PROPOSAL_VALIDATOR=critic \
bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
  oracle_planner \
  $(git rev-parse HEAD)

# 4. Run with custom settings
BLINDED_DEBUGGER_NUM_CANDIDATES=5 \
ORACLE_PLANNER_MAX_RETRIES=3 \
PROPOSAL_VALIDATOR=verifier \
bash evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh \
  oracle_planner \
  $(git rev-parse HEAD) \
  OracleTriadCodeActAgent \
  10         # eval limit (number of instances)
```

### Shell Launcher

**Script:** `evaluation/benchmarks/swe_bench_optimized/scripts/run_oracle_triad_infer.sh`

```
Usage:
  bash run_oracle_triad_infer.sh \
    <MODEL_CONFIG> <COMMIT_HASH> [AGENT] [EVAL_LIMIT] [MAX_ITER] \
    [NUM_WORKERS] [DATASET] [SPLIT] [N_RUNS]
```

| Argument | Position | Default | Description |
|----------|----------|---------|-------------|
| `MODEL_CONFIG` | 1 | *(required)* | LLM config name from config.toml |
| `COMMIT_HASH` | 2 | *(required)* | Git commit hash for version tracking |
| `AGENT` | 3 | `OracleTriadCodeActAgent` | Agent class name |
| `EVAL_LIMIT` | 4 | *(all instances)* | Max instances to evaluate |
| `MAX_ITER` | 5 | `100` | Max iterations per instance |
| `NUM_WORKERS` | 6 | `1` | Parallel evaluation workers |
| `DATASET` | 7 | `princeton-nlp/SWE-bench_Lite` | HuggingFace dataset |
| `SPLIT` | 8 | `test` | Dataset split |
| `N_RUNS` | 9 | `1` | Runs per instance |

All environment variables from the [Configuration](#environment-variables) section are accepted and exported by the script.

### Python Entry Point

**Script:** `evaluation/benchmarks/swe_bench_optimized/run_infer_oracle_triad.py`

Can also be invoked directly:

```bash
poetry run python evaluation/benchmarks/swe_bench_optimized/run_infer_oracle_triad.py \
  --agent-cls OracleTriadCodeActAgent \
  --llm-config oracle_planner \
  --max-iterations 100 \
  --eval-num-workers 1 \
  --eval-note "my-experiment" \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --n-runs 1
```

Optional flags: `--eval-n-limit <N>`, `--instance-ids <id1,id2,...>`

### Oracle Context Preparation

The planner requires an oracle context JSON file per instance. The eval runner creates this automatically from dataset columns, but you can also provide preprocessed analysis:

```bash
# Precomputed analysis directory structure:
ORACLE_PREPROCESS_DIR=/path/to/preprocess/
#   ├── django__django-12663_analysis.md      (deep analysis markdown)
#   └── django__django-12663_react_facts.json (structured investigation facts)
```

**React facts JSON format:**
```json
{
  "facts": [
    {
      "fact_id": "f1",
      "stage": "exploration",
      "goal": "Find the file containing the bug",
      "fact": "The bug is in django/core/management/commands/shell.py",
      "preconditions": ["The file exists in the repo"],
      "reasoning": "grep found the relevant code",
      "action": "grep -rn 'shell' django/core/management/",
      "observation": "Found match at line 42"
    }
  ]
}
```

---

## Step-by-Step Execution Flow

**Per step (one call to `agent.step()`):**

```
1. INITIALIZATION (once per episode, on first step)
   ├─ Load oracle context JSON (patch, test, analysis, react facts)
   ├─ Strip oracle blocks from issue text → public issue text
   ├─ Initialize ReactFactTracker
   ├─ Initialize OraclePlanner.from_env()
   └─ Initialize validator based on PROPOSAL_VALIDATOR:
      ├─ 'verifier' → HistoryGroundedVerifier.from_env()
      ├─ 'critic'   → OracleProposalCritic.from_env()
      └─ 'none'     → (neither)

2. BLIND CANDIDATE GENERATION
   └─ Call debugger LLM N times → N candidate response texts
      (same prompt, different completions)

3. PLANNING LOOP (up to max_retries + 1 iterations)
   ├─ Planner inspects history + candidates + oracle context
   ├─ Returns PlannerDecision:
   │   ├─ 'candidate' → use existing candidate, done
   │   └─ 'proposal'  → proposed new response text, needs validation
   │
   ├─ If proposal + validator:
   │   ├─ Validate proposal
   │   ├─ If valid → accept, done
   │   └─ If invalid → feed rejection reason back to planner, retry
   │
   └─ If retries exhausted → fallback to planner's best candidate

4. MATERIALIZATION (for accepted proposals)
   ├─ Inject planner guidance as user message to debugger LLM
   ├─ Call debugger LLM with guidance → generates tool calls + reasoning
   └─ Patch empty content if tool calls suppress reasoning

5. ACTION CONVERSION
   └─ response_to_actions() → queue actions, return first
```

---

## Prompt Templates

Located in `prompts/`:

| Template | Used By | Purpose |
|----------|---------|---------|
| `planner_select_or_propose.j2` | `OraclePlanner` | Ask planner to select a candidate or propose a revised response |
| `validate_oracle_proposal.j2` | `OracleProposalCritic` | Ask critic to judge proposal groundedness and leakage |
| `extract_claims.j2` | `ClaimExtractor` | Ask LLM to decompose proposal into structured claims and retrieval plan |
| `synthesize_verdict.j2` | `HistoryGroundedVerifier` | Ask LLM to resolve ambiguous leakage cases |

---

## Triad Log Format

Each step produces structured log entries saved to `{eval_output_dir}/oracle_triad_logs/{instance_id}.jsonl`:

```jsonl
{"step_index": 5, "event": "debugger_candidate", "candidate_index": 0, "response_text": "..."}
{"step_index": 5, "event": "debugger_candidate", "candidate_index": 1, "response_text": "..."}
{"step_index": 5, "event": "debugger_candidate", "candidate_index": 2, "response_text": "..."}
{"step_index": 5, "event": "oracle_planner_decision", "attempt": 0, "decision": "proposal", "reason": "...", ...}
{"step_index": 5, "event": "verifier_verdict", "attempt": 0, "verdict": "invalid", "reason": "...", "rule_results": [...]}
{"step_index": 5, "event": "oracle_planner_decision", "attempt": 1, "decision": "proposal", "reason": "...", ...}
{"step_index": 5, "event": "verifier_verdict", "attempt": 1, "verdict": "valid", ...}
{"step_index": 5, "event": "react_fact_usage_summary", "total_facts": 8, "used_facts": 3, ...}
```

The event type depends on the selected validator:
- `verifier_verdict` — when using `PROPOSAL_VALIDATOR=verifier`
- `proposal_critic_validation` — when using `PROPOSAL_VALIDATOR=critic`

---

## File Index

```
oracle_triad_codeact_agent/
├── __init__.py                     # Registers OracleTriadCodeActAgent
├── oracle_triad_codeact_agent.py   # Main agent: step loop, candidate generation, planner/validator orchestration
├── oracle_planner.py               # Oracle-aware planner: selects candidate or proposes revised action
├── proposal_critic.py              # Legacy one-shot LLM critic
├── verifier.py                     # 4-stage neuro-symbolic verifier (default validator)
├── history_memory.py               # Structured history indexing and bounded retrieval
├── claim_extractor.py              # LLM + programmatic claim/precondition extraction
├── symbolic_rules.py               # 14 deterministic rules across 5 families (A–E)
├── NEURO_SYMBOLIC_VERIFIER_DEV.md  # Verifier development plan, session logs
├── README.md                       # This file
└── prompts/
    ├── planner_select_or_propose.j2    # Planner decision prompt
    ├── validate_oracle_proposal.j2     # Legacy critic prompt
    ├── extract_claims.j2               # Claim extraction prompt (verifier Stage 1)
    └── synthesize_verdict.j2           # Verdict synthesis prompt (verifier Stage 4)
```
