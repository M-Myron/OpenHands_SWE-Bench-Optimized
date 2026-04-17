# Oracle-Guided SFT Data Distillation for Autonomous Software Engineering Agents

## Method Overview

We propose **Oracle-Guided Distillation**, a method for generating high-quality Supervised Fine-Tuning (SFT) trajectories from a weaker "student" model by using a privileged "oracle" supervisor that has access to the ground-truth solution. The key insight is that during trajectory generation, an oracle planner with access to the correct patch can guide a blinded solver along an optimal problem-solving path — producing training trajectories that teach the solver to follow expert-level debugging workflows, even though the solver itself never sees the solution.

## Problem Setting

**Autonomous software engineering** tasks require an agent to:
1. Read an issue description (bug report or feature request)
2. Explore a large codebase to understand the problem
3. Create reproduction scripts to confirm the bug
4. Analyze the root cause
5. Plan and implement a fix
6. Verify the fix passes tests

Current approaches either:
- **Distill from strong models** (GPT-4 → smaller model): expensive, quality limited by teacher capabilities
- **Use rejection sampling**: generate many trajectories from the student, keep successful ones. Very low success rate on hard tasks.
- **Reinforcement learning**: requires reward signals that are sparse (pass/fail) and expensive (running test suites)

Our approach uses **privileged information** (the known-correct patch) to guide the student during data generation, producing high-quality trajectories at much higher success rates.

## Three-Component Architecture

### Component 1: Blinded Solver (Student Model)

The blinded solver is the target model being distilled. It operates as a standard code agent:
- Receives the issue description and tool outputs
- Generates candidate responses (reasoning + tool calls)
- **Cannot see**: the ground-truth patch, the investigation fact graph, or the oracle's decisions

At each step, the solver generates N candidate responses. These candidates represent what the model would naturally produce. The oracle then selects the best candidate or modifies it.

### Component 2: Oracle Planner (Privileged Supervisor)

The oracle planner has access to:
- **The ground-truth patch** (diff showing exactly what code needs to change)
- **A structured investigation fact graph** (DAG of facts that must be discovered)
- **The solver's candidate responses** for the current step
- **The full interaction history** visible to the solver

The oracle makes one of three decisions per step:
- **Select**: pick the best candidate as-is (solver already on the right track)
- **Revise**: modify a candidate partially (close but needs adjustment)
- **Rewrite**: produce a completely new response (candidates are off-track)

Key constraint: **the oracle must never leak privileged information** into the response. The output response must read as if a skilled developer wrote it naturally — because this response becomes the SFT training target.

### Component 3: Hybrid Critic (Validation Layer)

When the oracle revises or rewrites a response, the critic validates it:

1. **Neural judgment**: An independent LLM (without oracle access) evaluates whether the response is grounded in the interaction history — checking for information leakage, unreachable file paths, or knowledge that couldn't have been obtained from previous steps.

2. **Symbolic regex checks**: The critic LLM extracts regex patterns that should match in the history (e.g., if the response references a file path, that path must appear somewhere in prior tool outputs). The regexes are tested programmatically.

3. **Realism check**: A programmatic scan detects leaked fact IDs, oracle terminology ("golden patch", "fact tracker"), or internal concepts that shouldn't appear in training data.

4. **Recheck on disagreement**: If neural says valid but symbolic rejects, a focused second LLM call determines whether the regex failure was a false positive.

If the critic rejects, the oracle receives feedback and retries with a different approach.

## Investigation Fact Graph

Each SWE-bench instance has a **pre-computed fact graph** — a directed acyclic graph (DAG) of knowledge nodes:

### Fact Nodes
Each fact is a discrete piece of knowledge about the codebase or bug:
```
f1: "The transact_write_items method has no duplicate item validation" (static)
f2: "MockValidationException is the base class for validation errors" (static)
f15: "Duplicate item operations currently succeed with HTTP 200" (dynamic)
```

Facts have:
- **type**: `static` (discovered by reading code) or `dynamic` (discovered by running code)
- **unlocker**: the action that reveals the fact (e.g., viewing a file, running a test)
- **depends_on**: prerequisite facts that must be unlocked first
- **statement**: what the fact says (used to verify the solver articulated the finding)

### Artifact Nodes
Artifacts represent structured outputs corresponding to debugging phases:
- `reproduce_script` — a reproduction test demonstrating the bug
- `issue_analysis` — root cause analysis (via reasoning or `think` tool)
- `fix_plan` — implementation plan
- `code_edit` — exact code changes (file, old_str, new_str)
- `validation` — running tests to verify

Artifacts have DAG dependencies that enforce a strict phase order:
```
facts → reproduce → analysis → plan → edits → validation
```

### Phase Gating

The system programmatically enforces that phases cannot be skipped:
- Cannot create a reproduction script until its dependency facts are unlocked
- Cannot enter analysis until reproduction is complete
- Cannot implement code changes until analysis and planning are done
- Cannot run validation until all edits are applied

This is enforced by detecting **tool actions** (not keywords) in the oracle's output:
- File creation → gates reproduction phase
- Code modification (`str_replace`, `sed -i`) → gates implementation phase
- Phase headers (`## Phase 5:`, `## Phase 6:`) → gates corresponding phase

## What Makes This Different

### vs. Standard Distillation (strong → weak)
- **Standard**: Teacher generates entire trajectory. Student learns to imitate.
- **Ours**: Student generates candidates. Oracle selects/corrects. Student's own distribution shapes the trajectory, with oracle providing minimal corrections. This reduces distribution shift between training and inference.

### vs. Rejection Sampling
- **Rejection sampling**: Generate 100 trajectories, keep the 2 that pass tests. Requires solving the problem independently. Very low yield on hard tasks.
- **Ours**: Oracle guides every step, achieving near-100% success rate on solvable problems. Each trajectory is high-quality by construction.

### vs. RLHF/Process Reward Models
- **RLHF**: Needs reward model training, reward hacking mitigation, PPO instability.
- **Ours**: No reward model needed. The oracle provides direct supervision at each step. The critic ensures quality. The result is SFT data — simple and stable to train on.

### The Leakage-Quality Trade-off
The central tension: the oracle knows the solution, but must guide the solver without revealing it. Too much intervention → leakage → model learns to expect hints that won't be there at inference. Too little intervention → solver takes inefficient paths → training data quality drops.

Our solution uses multiple enforcement layers:
1. **Prompt rules**: Oracle instructed never to mention fact IDs, golden patches, or oracle concepts
2. **Sanitizer**: Post-processing strips any leaked fact IDs from response text
3. **Critic neural check**: Independent LLM validates grounding in history
4. **Critic realism check**: Programmatic scan for oracle terminology
5. **Phase gating**: Prevents skipping investigation steps

## Fact Graph Construction (Pre-processing)

Fact graphs are pre-computed from the ground-truth patch using a two-stage pipeline:

**Stage 1**: An analysis LLM reads the issue + patch and produces a structured investigation plan — what facts would a developer need to discover, in what order, and what actions would reveal them.

**Stage 2**: A refinement LLM validates the graph, adds precise unlocker actions (exact file paths, line ranges, runnable test code), and ensures the DAG dependencies are correct.

The fact graph format (v6) uses a flat node list with typed nodes:
```json
{
  "nodes": [
    {"id": "f1", "node_type": "fact", "type": "static", "statement": "...", "depends_on": []},
    {"id": "repro1", "node_type": "reproduce_script", "code": "...", "depends_on": ["f1", "f15"]},
    {"id": "edit1", "node_type": "code_edit", "file": "...", "old_str": "...", "new_str": "...", "depends_on": ["plan"]}
  ]
}
```

## Training Data Format

Each step in the trajectory produces an SFT training example:
- **Input**: System prompt + conversation history up to the current step
- **Target**: The oracle-guided response (select/revise/rewrite output)

The target response looks identical to what a skilled developer would produce — it includes reasoning text followed by tool calls, with no traces of oracle guidance.

## Evaluation

The generated trajectories are evaluated on SWE-bench (Verified/Lite) by:
1. Fine-tuning the student model on the oracle-guided trajectories
2. Running the fine-tuned model on held-out instances (without oracle)
3. Measuring resolve rate (% of issues correctly fixed)

Key metrics during data generation:
- **Fact coverage**: % of investigation facts the solver unlocked during the trajectory
- **Phase completion**: whether all phases (repro → analysis → plan → edit → validate) were reached
- **Oracle intervention rate**: % of steps where oracle revised/rewrote vs. selected
- **Critic rejection rate**: % of oracle proposals rejected by the critic
- **Leakage incidents**: count of sanitizer/realism-check catches per trajectory
