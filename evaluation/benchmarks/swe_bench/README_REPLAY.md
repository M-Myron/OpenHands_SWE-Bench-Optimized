# Replay and Refine for SWE-bench

## What It Is

A tool to improve failed SWE-bench agent trajectories by:
1. **Replaying** the trajectory up to an error point
2. **Injecting** a corrective action
3. **Continuing** execution to generate a refined solution

The system restores the environment state (files, git, conversation) at the error point, allowing the agent to take a different action and potentially solve the task successfully.

## How to Use

### Quick Start

```bash
# Single trajectory refinement
python evaluation/benchmarks/swe_bench/replay_and_refine.py \
    --trajectory-path trajectory.json \
    --refinement-input refinement.json \
    --output-dir refined_output \
    --model-config llm

# Batch processing
python evaluation/benchmarks/swe_bench/replay_and_refine.py \
    --batch \
    --trajectory-dir trajectories/ \
    --refinement-inputs refinements.json \
    --output-dir refined_output \
    --model-config llm
```

### Prerequisites

- OpenHands environment with Docker
- Original trajectories from `run_infer.py` (saved in `output.jsonl`)
- Refinement input JSON specifying corrections

### Refinement Input Format

**Required fields:**
```json
{
  "instance_id": "django__django-12345",
  "target_step_id": 15,
  "suggested_action": {
    "action": "edit",
    "args": {
      "path": "/workspace/correct_file.py",
      "command": "view"
    }
  }
}
```

**Optional fields:**
```json
{
  "reason": "Agent edited wrong file",
  "bad_consequence": "Fix will be in wrong location"
}
```

### Supported Actions

**Edit file:**
```json
{"action": "edit", "args": {"path": "/workspace/file.py", "command": "view"}}
```

**Run command:**
```json
{"action": "run", "args": {"command": "pytest tests/test_file.py -xvs"}}
```

**Send message:**
```json
{"action": "message", "args": {"content": "Let's try a different approach..."}}
```

**Read file:**
```json
{"action": "read", "args": {"path": "/workspace/file.py"}}
```

**Finish:**
```json
{"action": "finish", "args": {"outputs": {}}}
```

### Complete Workflow

```bash
# 1. Run initial inference
python evaluation/benchmarks/swe_bench/run_infer.py \
    --agent-cls CodeActAgent \
    --llm-config llm \
    --max-iterations 30 \
    --eval-n-limit 10

# 2. Extract trajectories (already in output.jsonl)
python evaluation/benchmarks/swe_bench/extract_trajectories.py \
    --eval-output-dir ./evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/Qwen2.5-Coder-14B-Instruct_maxiter_100_N_v0.61.0-no-hint-run_1 \
    --output-dir ./evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/Qwen2.5-Coder-14B-Instruct_maxiter_100_N_v0.61.0-no-hint-run_1/trajectories/

# 3. Analyze and create refinement inputs (YOUR TOOL)
python your_analysis.py \
    --trajectory-dir trajectories/ \
    --output refinements.json

# 4. Run replay and refine
python evaluation/benchmarks/swe_bench/replay_and_refine.py \
    --batch \
    --trajectory-dir trajectories/ \
    --refinement-inputs refinements.json \
    --output-dir refined_output

# 5. Evaluate results
python evaluation/benchmarks/swe_bench/eval_infer.py \
    --output-dir refined_output
```

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--trajectory-path` | Yes (single) | - | Path to trajectory JSON |
| `--trajectory-dir` | Yes (batch) | - | Directory with trajectories |
| `--refinement-input` | Yes (single) | - | Refinement input JSON |
| `--refinement-inputs` | Yes (batch) | - | Batch refinement inputs JSON |
| `--output-dir` | Yes | - | Output directory |
| `--model-config` | No | llm | LLM config name |
| `--agent-class` | No | CodeActAgent | Agent class |
| `--max-iterations` | No | 50 | Max iterations |
| `--eval-note` | No | replay_refine | Evaluation tag |
| `--batch` | No | False | Enable batch mode |

## Examples

### Example 1: Wrong File Edited
```json
{
  "instance_id": "django__django-12345",
  "target_step_id": 15,
  "reason": "Agent edited models/base.py instead of models/query.py",
  "suggested_action": {
    "action": "edit",
    "args": {"path": "/workspace/django/db/models/query.py", "command": "view"}
  }
}
```

### Example 2: Wrong Test Command
```json
{
  "instance_id": "pytest__pytest-67890",
  "target_step_id": 8,
  "reason": "Agent ran all tests instead of specific failing test",
  "suggested_action": {
    "action": "run",
    "args": {"command": "pytest tests/test_specific.py::test_func -xvs"}
  }
}
```

### Example 3: Batch Refinements
```json
[
  {
    "instance_id": "django__django-12345",
    "target_step_id": 15,
    "suggested_action": {"action": "edit", "args": {"path": "/workspace/file1.py", "command": "view"}}
  },
  {
    "instance_id": "pytest__pytest-67890",
    "target_step_id": 8,
    "suggested_action": {"action": "run", "args": {"command": "pytest tests/test.py"}}
  }
]
```

## Key Points

- **Event IDs** start from 0; replay stops *before* `target_step_id`
- **Trajectories** are in `output.jsonl` from `run_infer.py` (under `history` field)
- **Output format** matches `run_infer.py` for direct evaluation with `eval_infer.py`
- **Environment state** is restored (files, git, conversation) at replay point
- **Non-deterministic actions** (network, time) may produce different results on replay

## Troubleshooting

**Replay doesn't restore state:**
- Verify same docker image and environment as original run

**Agent ignores suggested action:**
- Check action format matches supported types
- Choose earlier `target_step_id` if agent already finished

**Event ID not found:**
- Verify `target_step_id` exists in trajectory
- Event IDs start from 0

**Output evaluation fails:**
- Ensure `output.jsonl` has required fields: `instance_id`, `test_result`, `instance`
