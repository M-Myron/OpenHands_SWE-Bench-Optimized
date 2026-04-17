#!/usr/bin/env python3
"""Label rollout trajectory steps against ground-truth fact graphs.

Each agent step is classified as high_value / neutral / harmful using an LLM
judge. Results are saved per-instance as JSON files.

Usage:
    python label_trajectory.py \
        --output-jsonl /path/to/output.jsonl \
        --preprocess-dir /path/to/swegym_v6_phase1 \
        --instance-ids getmoto__moto-4787 getmoto__moto-4895 \
        --results-dir /path/to/labeled_trajectory_facts \
        --save-prompts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Any

from openai import OpenAI


# ---------------------------------------------------------------------------
# Step extraction
# ---------------------------------------------------------------------------

def extract_steps(history: list[dict]) -> list[dict]:
    """Extract agent action+observation pairs from raw history."""
    steps = []
    i = 0
    while i < len(history):
        event = history[i]
        action = event.get("action", "")
        source = event.get("source", "")

        if source == "agent" and action and action not in ("system",):
            step = {
                "step_num": len(steps),
                "event_index": i,
                "action_type": action,
                "action_event": event,
                "observation_event": None,
            }
            if i + 1 < len(history) and history[i + 1].get("observation"):
                step["observation_event"] = history[i + 1]
                i += 2
            else:
                i += 1
            steps.append(step)
        else:
            i += 1
    return steps


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

MAX_STEP_CONTENT_CHARS = 3000


def format_step_for_prompt(step: dict, max_chars: int = MAX_STEP_CONTENT_CHARS) -> str:
    action = step["action_event"]
    obs = step["observation_event"]
    action_type = step["action_type"]
    args = action.get("args", {})

    lines = [f"### Step {step['step_num']}  [action: {action_type}]"]

    thought = args.get("thought", "") or ""
    tool_meta = action.get("tool_call_metadata", {})
    model_resp = tool_meta.get("model_response", {}) if tool_meta else {}
    choices = model_resp.get("choices", []) if model_resp else []
    llm_content = ""
    if choices:
        msg = choices[0].get("message", {})
        llm_content = msg.get("content", "") or ""

    if llm_content and llm_content != thought:
        lines.append(f"**LLM reasoning**: {llm_content[:max_chars]}")
    elif thought:
        lines.append(f"**Thought**: {thought[:max_chars]}")

    if action_type == "run":
        lines.append(f"**Command**: `{args.get('command', '')[:max_chars]}`")
    elif action_type == "read":
        path = args.get("path", "")
        start = args.get("start_line", "")
        end = args.get("end_line", "")
        lines.append(f"**Read**: {path} (lines {start}-{end})")
    elif action_type == "edit":
        path = args.get("path", "")
        old_str = (args.get("old_str", "") or "")[:400]
        new_str = (args.get("new_str", "") or "")[:400]
        lines.append(f"**Edit**: {path}")
        if args.get("command") == "create":
            content = (args.get("file_text", "") or "")[:max_chars]
            lines.append(f"  create file: {content}")
        else:
            lines.append(f"  old: {old_str}")
            lines.append(f"  new: {new_str}")
    elif action_type == "think":
        lines.append(f"**Think**: {thought[:max_chars]}")
    elif action_type == "message":
        content = args.get("content", "")[:max_chars]
        lines.append(f"**Message**: {content}")
    elif action_type == "finish":
        lines.append("**Finish**: Agent declares task complete.")
    elif action_type == "run_ipython":
        code = args.get("code", "")[:max_chars]
        lines.append(f"**IPython**: {code}")
    else:
        lines.append(f"**Args**: {json.dumps(args)[:max_chars]}")

    if obs:
        obs_type = obs.get("observation", "")
        obs_content = (obs.get("content", "") or "")[:max_chars]
        extras = obs.get("extras", {})
        if obs_type == "run":
            metadata = extras.get("metadata", {})
            exit_code = metadata.get("exit_code", "?")
            lines.append(f"**Output** (exit_code={exit_code}): {obs_content}")
        elif obs_type == "read":
            lines.append(f"**File content**: {obs_content}")
        elif obs_type == "edit":
            lines.append(f"**Edit result**: {obs_content}")
        elif obs_type == "error":
            lines.append(f"**ERROR**: {obs_content}")
        elif obs_content:
            lines.append(f"**Observation** ({obs_type}): {obs_content}")

    return "\n".join(lines)


def format_fact_graph_for_prompt(nodes: list[dict]) -> str:
    lines = ["# Ground Truth Fact Graph\n"]
    lines.append("This is the oracle fact graph for this task. It represents the ideal ")
    lines.append("investigation path and correct fix. Use it to judge whether each step ")
    lines.append("in the trajectory aligns with or deviates from the ground truth.\n")

    facts = [n for n in nodes if n["node_type"] == "fact"]
    edits = [n for n in nodes if n["node_type"] == "code_edit"]
    repros = [n for n in nodes if n["node_type"] == "reproduce_script"]
    analyses = [n for n in nodes if n["node_type"] == "issue_analysis"]
    plans = [n for n in nodes if n["node_type"] == "fix_plan"]
    validations = [n for n in nodes if n["node_type"] == "validation"]
    others = [n for n in nodes if n["node_type"] not in
              ("fact", "code_edit", "reproduce_script", "issue_analysis", "fix_plan", "validation")]

    if facts:
        lines.append("\n## Investigation Facts")
        lines.append("Facts the agent should discover during exploration:\n")
        for f in facts:
            dep = f", depends_on={f['depends_on']}" if f.get("depends_on") else ""
            lines.append(f"- **[{f['id']}]** ({f.get('type','')}) {f['statement'][:300]}{dep}")
            unlocker = f.get("unlocker", {})
            if unlocker:
                lines.append(f"  - Discovery action: `{unlocker.get('action', '')}` → {unlocker.get('observation', '')[:200]}")

    if repros:
        lines.append("\n## Reproduction Scripts")
        for r in repros:
            lines.append(f"- **[{r['id']}]** {r.get('description', '')[:200]}")
            lines.append(f"  - Expected output before fix: {r.get('output_before_fix', '')[:200]}")
            lines.append(f"  - Expected output after fix: {r.get('output_after_fix', '')[:200]}")

    if analyses:
        lines.append("\n## Issue Analysis")
        for a in analyses:
            lines.append(f"- **[{a['id']}]** {a.get('text', '')[:400]}")

    if plans:
        lines.append("\n## Fix Plan")
        for p in plans:
            lines.append(f"- **[{p['id']}]** {p.get('text', '')[:400]}")

    if edits:
        lines.append("\n## Required Code Edits")
        lines.append("These are the exact code changes needed to fix the bug:\n")
        for e in edits:
            lines.append(f"- **[{e['id']}]** File: `{e.get('file', '')}` (action: {e.get('action_type', 'str_replace')})")
            old = (e.get("old_str", "") or "")[:300]
            new = (e.get("new_str", "") or "")[:300]
            if old:
                lines.append(f"  - Remove: ```{old}```")
            if new:
                lines.append(f"  - Insert: ```{new}```")

    if validations:
        lines.append("\n## Validation")
        for v in validations:
            lines.append(f"- **[{v['id']}]** Command: `{v.get('command', '')[:200]}`")
            lines.append(f"  - Expected: {v.get('expected_output', '')[:200]}")

    if others:
        lines.append("\n## Other Nodes")
        for o in others:
            lines.append(f"- **[{o['id']}]** ({o['node_type']}) {str(o.get('text', o.get('description', '')))[:200]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

LABELING_SYSTEM_PROMPT = """\
You are an expert code agent trajectory evaluator. Your task is to label each step
in an agent's problem-solving trajectory as high-value, neutral, or harmful for
supervised fine-tuning (SFT) data curation.

You are given:
1. A GROUND TRUTH FACT GRAPH: the ideal investigation path, correct analysis, and
   required code changes for solving a software bug.
2. A TRAJECTORY: the agent's actual step-by-step actions and observations.

For each step, assign ONE label and provide a brief reason:

## Labels

### "high_value" — Good SFT training signal
Steps that demonstrate desirable behavior:
- Targeted file exploration that aligns with facts in the ground truth
- Correct reproduction of the bug as described in the fact graph
- Accurate root cause analysis matching the ground truth
- Code edits that match or approximate the required fixes
- Efficient, well-scoped validation of the fix
- Well-reasoned thinking/planning that progresses toward the solution
- Actions that directly discover facts listed in the fact graph

### "neutral" — Low signal, harmless boilerplate
Steps that are routine and neither helpful nor harmful:
- Standard environment setup (cd, ls, pwd)
- Reading the problem statement (first time only)
- Generic project structure exploration (if brief)
- Acknowledging task completion (finish action)

### "harmful" — Bad SFT training signal
Steps that teach the model undesirable behavior:
- **wrong_direction**: Investigating files/code unrelated to the ground truth fix
- **redundant_exploration**: Re-reading files already read, re-running same commands
- **information_less**: Commands that produce no useful information (e.g., grepping
  for irrelevant terms, reading unrelated files)
- **tool_misuse**: Using wrong tool for the job, malformed commands, syntax errors
- **incorrect_edit**: Code changes that don't match the ground truth and introduce
  bugs or are in wrong files
- **excessive_validation**: Over-long, repetitive test runs beyond what's needed
  to verify the fix
- **hallucination**: Agent claims something about the code that contradicts the
  fact graph or observed output
- **premature_action**: Implementing a fix before understanding the problem
  (skipping exploration/analysis)
- **backtracking**: Undoing correct work or reverting to a worse state

## Output Format

Return a JSON array where each element corresponds to a step:

```json
[
  {
    "step": 0,
    "label": "high_value",
    "sub_label": "",
    "reason": "Reads the exact file containing the bug as indicated by fact f3.",
    "related_facts": ["f3"]
  },
  {
    "step": 1,
    "label": "harmful",
    "sub_label": "redundant_exploration",
    "reason": "Re-reads the same file that was already fully read in step 0.",
    "related_facts": []
  }
]
```

Rules:
- `sub_label` is required for "harmful" steps (one of: wrong_direction,
  redundant_exploration, information_less, tool_misuse, incorrect_edit,
  excessive_validation, hallucination, premature_action, backtracking).
  For "high_value" and "neutral" steps, set sub_label to "".
- `related_facts` lists fact graph node IDs (e.g., ["f1", "f3", "edit1"]) that
  the step relates to. Empty list if unrelated.
- `reason` should be 1-2 sentences max.
- Be precise: a step exploring a file IS high_value if that file is mentioned in
  the fact graph, even if the agent doesn't explicitly cite the fact.
- Consider CONTEXT: a step may be neutral on its own but harmful if it's the 3rd
  time the agent reads the same file.
- Return ONLY the JSON array, no other text.
"""


# ---------------------------------------------------------------------------
# LLM labeling
# ---------------------------------------------------------------------------

def build_labeling_prompt(
    fact_graph_text: str,
    steps: list[dict],
    batch_start: int,
    batch_end: int,
    previous_labels: list[dict] | None = None,
) -> str:
    parts = [fact_graph_text, "\n\n---\n\n"]

    if previous_labels:
        parts.append("# Previously Labeled Steps (for context)\n")
        for lab in previous_labels[-10:]:
            parts.append(
                f"- Step {lab['step']}: **{lab['label']}** "
                f"{'(' + lab['sub_label'] + ') ' if lab.get('sub_label') else ''}"
                f"— {lab['reason']}\n"
            )
        parts.append("\n---\n\n")

    parts.append("# Trajectory Steps to Label\n\n")
    for s in steps[batch_start:batch_end]:
        parts.append(format_step_for_prompt(s))
        parts.append("\n\n")

    parts.append(
        f"\nLabel steps {batch_start} through {batch_end - 1}. "
        f"Return a JSON array with {batch_end - batch_start} elements."
    )
    return "".join(parts)


def call_llm_for_labels(
    client: OpenAI,
    model: str,
    fact_graph_text: str,
    steps: list[dict],
    batch_start: int,
    batch_end: int,
    previous_labels: list[dict] | None = None,
    max_retries: int = 6,
    prompt_log: list[dict] | None = None,
) -> list[dict]:
    prompt = build_labeling_prompt(
        fact_graph_text, steps, batch_start, batch_end, previous_labels
    )

    raw = ""
    for attempt in range(max_retries):
        backoff = min(2 ** attempt, 60)  # 1, 2, 4, 8, 16, 60

        # ── LLM call with connection error handling ──
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": LABELING_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
                timeout=300,
            )
        except Exception as e:
            err_name = type(e).__name__
            print(f"    Connection error on attempt {attempt + 1}/{max_retries} ({err_name}): {e}")
            if attempt < max_retries - 1:
                print(f"    Retrying in {backoff}s...")
                time.sleep(backoff)
            continue

        # ── Validate response structure ──
        try:
            if not resp.choices:
                print(f"    Empty choices on attempt {attempt + 1}/{max_retries}. Retrying in {backoff}s...")
                time.sleep(backoff)
                continue
            content = resp.choices[0].message.content
            if not content or not content.strip():
                print(f"    Empty response on attempt {attempt + 1}/{max_retries}. Retrying in {backoff}s...")
                time.sleep(backoff)
                continue
            raw = content.strip()
        except (AttributeError, IndexError) as e:
            print(f"    Malformed response on attempt {attempt + 1}/{max_retries}: {e}. Retrying in {backoff}s...")
            time.sleep(backoff)
            continue

        # Log prompt/response if requested
        if prompt_log is not None:
            prompt_log.append({
                "batch_start": batch_start,
                "batch_end": batch_end,
                "attempt": attempt,
                "prompt_chars": len(prompt),
                "prompt": prompt,
                "response": raw,
                "usage": {
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                },
            })

        # ── Parse JSON from response ──
        try:
            parse_target = raw
            # Extract JSON from markdown code blocks
            if parse_target.startswith("```"):
                lines = parse_target.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.strip().startswith("```") and not in_block:
                        in_block = True
                        continue
                    elif line.strip() == "```" and in_block:
                        break
                    elif in_block:
                        json_lines.append(line)
                parse_target = "\n".join(json_lines)

            # Try to find JSON array if response has extra text
            if not parse_target.strip().startswith("["):
                bracket_start = parse_target.find("[")
                bracket_end = parse_target.rfind("]")
                if bracket_start != -1 and bracket_end > bracket_start:
                    parse_target = parse_target[bracket_start : bracket_end + 1]

            labels = json.loads(parse_target)
        except json.JSONDecodeError as e:
            print(f"    JSON parse error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                print(f"    Raw response: {raw[:500]}")
            elif attempt < max_retries - 1:
                time.sleep(backoff)
            continue

        # ── Validate label count ──
        expected_count = batch_end - batch_start
        if not isinstance(labels, list):
            print(f"    Response is not a list on attempt {attempt + 1}/{max_retries}. Retrying...")
            time.sleep(backoff)
            continue

        if len(labels) != expected_count:
            print(f"    Warning: expected {expected_count} labels, got {len(labels)}. Retrying...")
            time.sleep(backoff)
            continue

        # ── Validate label values ──
        valid_labels = {"high_value", "neutral", "harmful"}
        all_valid = True
        for lab in labels:
            if not isinstance(lab, dict):
                print(f"    Warning: label entry is not a dict. Retrying...")
                all_valid = False
                break
            if lab.get("label") not in valid_labels:
                print(f"    Warning: invalid label '{lab.get('label')}'. Retrying...")
                all_valid = False
                break
        if all_valid:
            return labels
        time.sleep(backoff)

    print(f"    FAILED to get labels for steps {batch_start}-{batch_end - 1} after {max_retries} attempts")
    return [
        {"step": i, "label": "unknown", "sub_label": "", "reason": "LLM labeling failed", "related_facts": []}
        for i in range(batch_start, batch_end)
    ]


# ---------------------------------------------------------------------------
# Per-instance processing
# ---------------------------------------------------------------------------

def process_instance(
    instance_id: str,
    output_jsonl: str,
    byte_offset: int,
    preprocess_dir: str,
    results_dir: str,
    client: OpenAI,
    model: str,
    steps_per_batch: int = 30,
    save_prompts: bool = False,
) -> dict:
    """Label all steps for a single instance. Returns summary dict."""
    print(f"\n{'='*70}")
    print(f"Processing: {instance_id}")
    print(f"{'='*70}")
    t_start = time.time()

    # Load trajectory
    with open(output_jsonl, "r") as f:
        f.seek(byte_offset)
        line = f.readline()
        output_entry = json.loads(line)

    history = output_entry["history"]
    steps = extract_steps(history)
    print(f"  Trajectory: {len(history)} events → {len(steps)} agent steps")

    step_types = Counter(s["action_type"] for s in steps)
    for k, v in step_types.most_common():
        print(f"    {k}: {v}")

    # Load fact graph
    facts_path = os.path.join(preprocess_dir, instance_id, "stage2_facts.json")
    with open(facts_path, "r") as f:
        fact_graph = json.load(f)
    nodes = fact_graph["nodes"]
    fact_nodes = [n for n in nodes if n["node_type"] == "fact"]
    print(f"  Fact graph: {len(nodes)} nodes ({len(fact_nodes)} facts)")

    fact_graph_text = format_fact_graph_for_prompt(nodes)

    # Run labeling
    prompt_log: list[dict] | None = [] if save_prompts else None
    all_labels: list[dict] = []
    n_steps = len(steps)
    n_batches = (n_steps + steps_per_batch - 1) // steps_per_batch

    print(f"  Labeling {n_steps} steps in {n_batches} batches...")

    for batch_idx in range(n_batches):
        batch_start = batch_idx * steps_per_batch
        batch_end = min(batch_start + steps_per_batch, n_steps)

        print(f"    Batch {batch_idx + 1}/{n_batches}: steps {batch_start}-{batch_end - 1} ... ", end="", flush=True)
        t0 = time.time()

        batch_labels = call_llm_for_labels(
            client, model, fact_graph_text, steps,
            batch_start, batch_end,
            previous_labels=all_labels if all_labels else None,
            prompt_log=prompt_log,
        )

        for i, lab in enumerate(batch_labels):
            lab["step"] = batch_start + i

        all_labels.extend(batch_labels)
        elapsed = time.time() - t0
        label_summary = Counter(l["label"] for l in batch_labels)
        print(f"done ({elapsed:.1f}s) — {dict(label_summary)}")

    # Compute summary
    label_counts = Counter(l["label"] for l in all_labels)
    harmful_subs = Counter(
        l.get("sub_label", "") for l in all_labels if l["label"] == "harmful"
    )
    high_value_count = label_counts.get("high_value", 0)
    harmful_count = label_counts.get("harmful", 0)
    sft_quality = high_value_count / n_steps if n_steps > 0 else 0
    harmful_ratio = harmful_count / n_steps if n_steps > 0 else 0

    # Fact coverage
    all_fact_ids = set(n["id"] for n in nodes)
    covered_facts = set()
    for lab in all_labels:
        if lab["label"] == "high_value":
            for fid in lab.get("related_facts", []):
                if fid in all_fact_ids:
                    covered_facts.add(fid)

    summary = {
        "instance_id": instance_id,
        "total_steps": n_steps,
        "high_value": high_value_count,
        "neutral": label_counts.get("neutral", 0),
        "harmful": harmful_count,
        "unknown": label_counts.get("unknown", 0),
        "sft_quality_score": round(sft_quality, 4),
        "harmful_ratio": round(harmful_ratio, 4),
        "harmful_sub_labels": dict(harmful_subs),
        "fact_coverage": f"{len(covered_facts)}/{len(all_fact_ids)}",
        "elapsed_seconds": round(time.time() - t_start, 1),
    }

    print(f"\n  Summary: {summary}")

    # Save results
    labels_dir = os.path.join(results_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    result_data = {
        "instance_id": instance_id,
        "model": model,
        "total_steps": n_steps,
        "summary": summary,
        "labels": all_labels,
    }

    result_path = os.path.join(labels_dir, f"{instance_id}.json")
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"  Saved labels: {result_path}")

    # Save prompts/responses if enabled
    if save_prompts and prompt_log:
        prompts_dir = os.path.join(results_dir, "prompts")
        os.makedirs(prompts_dir, exist_ok=True)
        prompt_path = os.path.join(prompts_dir, f"{instance_id}.jsonl")
        with open(prompt_path, "w") as f:
            for entry in prompt_log:
                f.write(json.dumps(entry) + "\n")
        print(f"  Saved prompts: {prompt_path}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Label trajectory steps against fact graphs using LLM judge"
    )
    parser.add_argument(
        "--output-jsonl", required=True,
        help="Path to output.jsonl with trajectories",
    )
    parser.add_argument(
        "--preprocess-dir", required=True,
        help="Path to preprocessed fact graphs (contains {instance_id}/stage2_facts.json)",
    )
    parser.add_argument(
        "--instance-ids", nargs="+", default=None,
        help="Instance IDs to process (default: all available)",
    )
    parser.add_argument(
        "--max-instances", type=int, default=None,
        help="Cap the number of instances to process",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Directory to save results (default: <output_jsonl_dir>/labeled_trajectory_facts/)",
    )
    parser.add_argument(
        "--llm-base-url", default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible LLM endpoint",
    )
    parser.add_argument(
        "--llm-model", default="zai-org/GLM-5-FP8",
        help="Model name",
    )
    parser.add_argument(
        "--steps-per-batch", type=int, default=30,
        help="Steps to send per LLM call",
    )
    parser.add_argument(
        "--save-prompts", action="store_true",
        help="Save intermediate prompts and responses for debugging",
    )
    args = parser.parse_args()

    # Default results dir alongside the output.jsonl
    if args.results_dir is None:
        output_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
        args.results_dir = os.path.join(output_dir, "labeled_trajectory_facts")

    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Results dir: {args.results_dir}")
    print(f"Save prompts: {args.save_prompts}")

    # Build instance index
    print(f"\nIndexing {args.output_jsonl} ...")
    instance_index: dict[str, int] = {}
    with open(args.output_jsonl, "r") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                entry = json.loads(line)
                iid = entry.get("instance_id", "")
                if iid:
                    instance_index[iid] = offset
            except json.JSONDecodeError:
                continue

    print(f"  Found {len(instance_index)} instances in output.jsonl")

    # Find instances with fact graphs
    instances_with_facts = set()
    for entry in os.listdir(args.preprocess_dir):
        if os.path.isfile(os.path.join(args.preprocess_dir, entry, "stage2_facts.json")):
            instances_with_facts.add(entry)

    available = sorted(set(instance_index.keys()) & instances_with_facts)
    print(f"  Instances with both trajectory + fact graph: {len(available)}")

    # Filter by requested IDs
    if args.instance_ids:
        requested = set(args.instance_ids)
        missing = requested - set(available)
        if missing:
            print(f"  WARNING: not found: {missing}")
        target_ids = [iid for iid in available if iid in requested]
    else:
        target_ids = available

    # Skip already-labeled instances
    labels_dir = os.path.join(args.results_dir, "labels")
    already_done = set()
    if os.path.isdir(labels_dir):
        for f in os.listdir(labels_dir):
            if f.endswith(".json"):
                already_done.add(f.replace(".json", ""))
    if already_done:
        before = len(target_ids)
        target_ids = [iid for iid in target_ids if iid not in already_done]
        print(f"  Skipping {before - len(target_ids)} already-labeled instances")

    if args.max_instances:
        target_ids = target_ids[: args.max_instances]

    print(f"\nWill process {len(target_ids)} instances")

    if not target_ids:
        print("Nothing to do.")
        return

    # LLM client
    client = OpenAI(base_url=args.llm_base_url, api_key="dummy")

    # Process instances
    all_summaries: list[dict] = []
    for idx, iid in enumerate(target_ids):
        print(f"\n[{idx + 1}/{len(target_ids)}]", end="")
        try:
            summary = process_instance(
                instance_id=iid,
                output_jsonl=args.output_jsonl,
                byte_offset=instance_index[iid],
                preprocess_dir=args.preprocess_dir,
                results_dir=args.results_dir,
                client=client,
                model=args.llm_model,
                steps_per_batch=args.steps_per_batch,
                save_prompts=args.save_prompts,
            )
            all_summaries.append(summary)
        except Exception as e:
            print(f"\n  ERROR processing {iid}: {e}")
            import traceback
            traceback.print_exc()
            all_summaries.append({
                "instance_id": iid,
                "error": str(e),
            })

    # Save aggregate summary
    agg_path = os.path.join(args.results_dir, "summary.jsonl")
    with open(agg_path, "a") as f:
        for s in all_summaries:
            f.write(json.dumps(s) + "\n")
    print(f"\nAppended {len(all_summaries)} summaries to: {agg_path}")

    # Print aggregate stats
    completed = [s for s in all_summaries if "error" not in s]
    if completed:
        avg_sft = sum(s["sft_quality_score"] for s in completed) / len(completed)
        avg_harmful = sum(s["harmful_ratio"] for s in completed) / len(completed)
        print(f"\n{'='*70}")
        print(f"AGGREGATE ({len(completed)} instances):")
        print(f"  Mean SFT quality:  {avg_sft:.1%}")
        print(f"  Mean harmful rate: {avg_harmful:.1%}")
        print(f"  Errors: {len(all_summaries) - len(completed)}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
