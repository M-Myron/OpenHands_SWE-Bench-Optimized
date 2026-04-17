"""LLM-assisted fact-graph reduction.

For each remaining (non-filtered) instance with non-problem-statement root
nodes, this script:

1. Extracts the fact-only subgraph reachable from each non-PS root.
2. Asks the LLM whether the subgraph is completely unrelated to the problem
   statement.
3. If unrelated, removes those fact nodes and rewrites/removes downstream
   artifact nodes (analysis, plan, edit, validation) that depend on them.
4. Saves the reduced graph to a new output directory.

Usage:
    python -m evaluation.benchmarks.swe_bench_optimized.preprocess.reduce_fact_graphs \
        --preprocess-dir <DIR> \
        --filter-json <FILTER_JSON> \
        --output-dir <DIR> \
        [--api-base http://localhost:8000/v1] \
        [--model glm-5] \
        [--dataset SWE-Gym/SWE-Gym] \
        [--split train] \
        [--max-workers 4] \
        [--dry-run]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import openai
from datasets import load_dataset


# ── Debug logger ─────────────────────────────────────────────────────────

class DebugLogger:
    """Thread-safe per-instance JSONL logger for LLM prompts/responses."""

    def __init__(self, log_dir: str | None) -> None:
        self._log_dir = log_dir
        self._lock = threading.Lock()
        self._counter: dict[str, int] = {}  # instance_id -> call seq
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._log_dir is not None

    def log(self, instance_id: str, call_type: str,
            system: str, user: str, response: str, extra: dict | None = None) -> None:
        if not self._log_dir:
            return
        with self._lock:
            seq = self._counter.get(instance_id, 0)
            self._counter[instance_id] = seq + 1
        entry = {
            'seq': seq,
            'call_type': call_type,
            'instance_id': instance_id,
            'system_prompt': system,
            'user_prompt': user,
            'response': response,
        }
        if extra:
            entry.update(extra)
        log_path = os.path.join(self._log_dir, f'{instance_id}.jsonl')
        line = json.dumps(entry, ensure_ascii=False) + '\n'
        with self._lock:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(line)


# ── Node type constants ──────────────────────────────────────────────────
_FACT = 'fact'
_ARTIFACT_TYPES = {'reproduce_script', 'issue_analysis', 'fix_plan', 'code_edit', 'validation'}
_REWRITABLE_TYPES = {'issue_analysis', 'fix_plan'}
_REWRITE_OR_REMOVE_TYPES = {'code_edit', 'validation', 'reproduce_script'}


# ── Helpers ──────────────────────────────────────────────────────────────

def _classify_roots(nodes: list[dict]) -> tuple[list[str], list[str]]:
    """Return (ps_root_ids, non_ps_root_ids)."""
    ps, non_ps = [], []
    for n in nodes:
        if n.get('depends_on'):
            continue
        unlocker = n.get('unlocker', {})
        action = unlocker.get('action', '') if isinstance(unlocker, dict) else ''
        if 'problem_statement' in action.lower():
            ps.append(n['id'])
        else:
            non_ps.append(n['id'])
    return ps, non_ps


def _build_maps(nodes: list[dict]) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Return (id→node, parent→children)."""
    by_id = {n['id']: n for n in nodes}
    children: dict[str, list[str]] = {}
    for n in nodes:
        for dep in n.get('depends_on', []):
            children.setdefault(dep, []).append(n['id'])
    return by_id, children


def _get_descendants(root_id: str, children: dict[str, list[str]]) -> set[str]:
    """BFS to get all descendant IDs (including root)."""
    visited: set[str] = set()
    queue = deque([root_id])
    while queue:
        nid = queue.popleft()
        if nid in visited:
            continue
        visited.add(nid)
        for child in children.get(nid, []):
            queue.append(child)
    return visited


def _extract_fact_subgraph(
    root_id: str,
    by_id: dict[str, dict],
    children: dict[str, list[str]],
) -> list[dict]:
    """Extract fact-only nodes reachable from root (BFS, only following fact→fact edges)."""
    fact_nodes: list[dict] = []
    visited: set[str] = set()
    queue = deque([root_id])
    while queue:
        nid = queue.popleft()
        if nid in visited:
            continue
        visited.add(nid)
        node = by_id.get(nid)
        if not node:
            continue
        if node['node_type'] != _FACT:
            continue
        fact_nodes.append(node)
        for child_id in children.get(nid, []):
            child = by_id.get(child_id)
            if child and child['node_type'] == _FACT:
                queue.append(child_id)
    return fact_nodes


def _find_affected_artifacts(
    removed_fact_ids: set[str],
    by_id: dict[str, dict],
    children: dict[str, list[str]],
) -> list[dict]:
    """Find artifact nodes whose depends_on includes any removed fact."""
    affected: list[dict] = []
    seen: set[str] = set()
    for fid in removed_fact_ids:
        for child_id in children.get(fid, []):
            if child_id in seen or child_id in removed_fact_ids:
                continue
            seen.add(child_id)
            node = by_id.get(child_id)
            if node and node['node_type'] in _ARTIFACT_TYPES:
                affected.append(node)
    return affected


# ── View-chain merge helpers ─────────────────────────────────────────────

def _parse_view_action(action: str) -> tuple[str, int, int] | None:
    """Parse ``[view] filepath start-end`` → (filepath, start, end) or None."""
    m = re.match(r'\[view\]\s+(\S+)\s+(\d+)\s*[-–]\s*(\d+)', action)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    m = re.match(r'\[view\]\s+(\S+)\s+(\d+)$', action)
    if m:
        line = int(m.group(2))
        return m.group(1), line, line
    return None


def _ranges_overlap_or_near(
    s1: int, e1: int, s2: int, e2: int, gap_tolerance: int = 0,
) -> bool:
    """True if [s1,e1] and [s2,e2] overlap, contain, or are within gap_tolerance lines."""
    return s1 <= e2 + gap_tolerance and s2 <= e1 + gap_tolerance


_DEFAULT_VIEW_MERGE_GAP = 50


def _merge_view_nodes(
    nodes: list[dict], gap_tolerance: int = _DEFAULT_VIEW_MERGE_GAP,
) -> tuple[list[dict], dict]:
    """Merge static fact nodes that view overlapping/contained ranges of the
    same file into their parent node.

    A child is absorbed into its parent when:
    - Both are static facts with ``[view]`` unlockers on the same file.
    - The child's view range overlaps with, is contained in, or is within
      ``gap_tolerance`` lines of the parent's range.
    - The child depends ONLY on the parent (single dependency).

    After absorption the parent's range widens to cover both, statements and
    observations are concatenated, and all of the child's dependants are
    re-parented to the parent.  The process repeats until no more merges are
    possible.

    Returns ``(new_nodes, stats_dict)``.
    """
    by_id = {n['id']: n for n in nodes}

    # Identify static view-facts
    view_info: dict[str, tuple[str, int, int]] = {}
    for n in nodes:
        if n.get('node_type') != _FACT or n.get('type') != 'static':
            continue
        unlocker = n.get('unlocker', {})
        action = unlocker.get('action', '') if isinstance(unlocker, dict) else ''
        parsed = _parse_view_action(action)
        if parsed:
            view_info[n['id']] = parsed

    if not view_info:
        return nodes, {'merges': 0, 'nodes_merged': 0}

    nodes_to_remove: set[str] = set()
    id_remap: dict[str, str] = {}  # absorbed_id → target_id
    total_merges = 0

    # Iterate until stable — a merge may enable further merges
    changed = True
    while changed:
        changed = False

        # Rebuild children map each round (deps change after merges)
        children_map: dict[str, list[str]] = {}
        for n in by_id.values():
            if n['id'] in nodes_to_remove:
                continue
            for dep in n.get('depends_on', []):
                children_map.setdefault(dep, []).append(n['id'])

        for nid in list(view_info):
            if nid in nodes_to_remove:
                continue
            node = by_id[nid]
            deps = node.get('depends_on', [])
            if len(deps) != 1:
                continue
            parent_id = deps[0]
            if parent_id in nodes_to_remove:
                continue
            if parent_id not in view_info:
                continue
            p_file, p_start, p_end = view_info[parent_id]
            c_file, c_start, c_end = view_info[nid]
            if p_file != c_file:
                continue
            if not _ranges_overlap_or_near(p_start, p_end, c_start, c_end, gap_tolerance):
                continue

            # Merge child into parent
            parent = by_id[parent_id]
            # Widen range
            new_start = min(p_start, c_start)
            new_end = max(p_end, c_end)
            view_info[parent_id] = (p_file, new_start, new_end)

            unlocker = parent.get('unlocker', {})
            unlocker['action'] = f'[view] {p_file} {new_start}-{new_end}'
            child_obs = node.get('unlocker', {}).get('observation', '')
            if child_obs:
                parent_obs = unlocker.get('observation', '')
                unlocker['observation'] = (
                    f'{parent_obs}\n{child_obs}' if parent_obs else child_obs
                )
            parent['unlocker'] = unlocker

            child_stmt = node.get('statement', '')
            if child_stmt:
                parent_stmt = parent.get('statement', '')
                parent['statement'] = (
                    f'{parent_stmt} {child_stmt}' if parent_stmt else child_stmt
                )

            by_id[parent_id] = parent
            nodes_to_remove.add(nid)
            id_remap[nid] = parent_id

            # Re-parent the child's children
            for grandchild_id in children_map.get(nid, []):
                if grandchild_id in nodes_to_remove:
                    continue
                gc = by_id[grandchild_id]
                gc['depends_on'] = [
                    parent_id if d == nid else d
                    for d in gc.get('depends_on', [])
                ]
                # De-dup
                seen: list[str] = []
                for d in gc['depends_on']:
                    if d not in seen:
                        seen.append(d)
                gc['depends_on'] = seen

            total_merges += 1
            changed = True

    if not nodes_to_remove:
        return nodes, {'merges': 0, 'nodes_merged': 0}

    # Resolve transitive remaps (A→B→C ⇒ A→C)
    def _resolve(nid: str) -> str:
        visited: set[str] = set()
        while nid in id_remap and nid not in visited:
            visited.add(nid)
            nid = id_remap[nid]
        return nid

    # Rebuild node list with updated references
    new_nodes: list[dict] = []
    for node in nodes:
        if node['id'] in nodes_to_remove:
            continue
        n = by_id.get(node['id'], node)
        updated = copy.deepcopy(n)
        new_deps: list[str] = []
        for d in updated.get('depends_on', []):
            mapped = _resolve(d)
            if mapped not in new_deps:
                new_deps.append(mapped)
        updated['depends_on'] = new_deps
        new_nodes.append(updated)

    stats = {
        'merges': total_merges,
        'nodes_merged': len(nodes_to_remove),
    }
    return new_nodes, stats


def _remove_facts_and_update_deps(
    nodes: list[dict],
    remove_ids: set[str],
) -> list[dict]:
    """Remove fact nodes and rewire dependencies to maintain connectivity.

    Children of a removed node inherit its parents (``depends_on``).
    Handles transitive removal (walks up through multiple removed nodes).
    """
    if not remove_ids:
        return nodes

    by_id = {n['id']: n for n in nodes}

    def _resolve_parents(nid: str, seen: set[str] | None = None) -> list[str]:
        if seen is None:
            seen = set()
        if nid in seen:
            return []
        seen.add(nid)
        result: list[str] = []
        for p in by_id.get(nid, {}).get('depends_on', []):
            if p in remove_ids:
                result.extend(_resolve_parents(p, seen))
            else:
                result.append(p)
        return result

    resolved = {nid: list(dict.fromkeys(_resolve_parents(nid))) for nid in remove_ids}

    new_nodes: list[dict] = []
    for node in nodes:
        if node['id'] in remove_ids:
            continue
        updated = copy.deepcopy(node)
        new_deps: list[str] = []
        for d in updated.get('depends_on', []):
            if d in remove_ids:
                for p in resolved.get(d, []):
                    if p not in new_deps:
                        new_deps.append(p)
            else:
                if d not in new_deps:
                    new_deps.append(d)
        updated['depends_on'] = new_deps
        new_nodes.append(updated)
    return new_nodes


# ── LLM calls ───────────────────────────────────────────────────────────

def _llm_call(
    client: openai.OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    debug_logger: DebugLogger | None = None,
    debug_instance_id: str = '',
    debug_call_type: str = '',
) -> str:
    """Call the LLM and return the response text.

    Retries on empty/None responses and API/connection errors (up to 5 attempts
    with exponential backoff: 1s, 2s, 4s, 8s, 16s).
    """
    max_attempts = 5
    t0 = time.time()
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = (resp.choices[0].message.content or '').strip()
            if not content:
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise ValueError(
                    f'LLM returned empty response after {max_attempts} attempts'
                )
            elapsed = time.time() - t0
            if debug_logger and debug_logger.enabled:
                debug_logger.log(
                    debug_instance_id, debug_call_type,
                    system, user, content,
                    extra={'attempt': attempt, 'elapsed_s': round(elapsed, 2)},
                )
            return content
        except Exception as e:
            if attempt < max_attempts - 1:
                wait = 2 ** attempt
                print(f'  [retry {attempt+1}/{max_attempts}] {type(e).__name__}: {e} '
                      f'(waiting {wait}s)')
                time.sleep(wait)
                continue
            raise


def _ask_relevance(
    client: openai.OpenAI,
    model: str,
    problem_statement: str,
    fact_nodes: list[dict],
    debug_logger: DebugLogger | None = None,
    debug_instance_id: str = '',
) -> bool:
    """Ask LLM if the fact subgraph is completely unrelated to the problem.

    Returns True if UNRELATED (should be removed).
    """
    facts_text = '\n'.join(
        f"- [{n['id']}] {n.get('statement', '(no statement)')}"
        for n in fact_nodes
    )
    system = (
        "You are a software engineering expert. You will be given a problem statement "
        "and a set of facts discovered during code investigation. Your task is to "
        "determine if these facts are COMPLETELY UNRELATED to the problem described "
        "in the problem statement.\n\n"
        "A fact subgraph is 'completely unrelated' if:\n"
        "- It describes a different feature, bug, or area of the codebase\n"
        "- None of the facts provide context, background, or prerequisite knowledge "
        "for understanding or fixing the reported problem\n"
        "- Removing these facts would not reduce the solver's ability to fix the problem\n\n"
        "Respond with EXACTLY one of:\n"
        "- UNRELATED: if the facts are completely unrelated to the problem\n"
        "- RELATED: if any of the facts are relevant to the problem\n\n"
        "Then provide a brief one-sentence justification."
    )
    user = (
        f"## Problem Statement\n{problem_statement}\n\n"
        f"## Fact Subgraph\n{facts_text}"
    )
    response = _llm_call(
        client, model, system, user,
        debug_logger=debug_logger,
        debug_instance_id=debug_instance_id,
        debug_call_type='relevance',
    )
    # Parse the verdict
    first_line = response.split('\n')[0].strip().upper()
    return 'UNRELATED' in first_line


def _ask_rewrite_artifact(
    client: openai.OpenAI,
    model: str,
    problem_statement: str,
    artifact_node: dict,
    removed_fact_statements: list[str],
    debug_logger: DebugLogger | None = None,
    debug_instance_id: str = '',
) -> dict | None:
    """Ask LLM to rewrite or remove an artifact node.

    For analysis/plan nodes: rewrite to remove unrelated parts.
    For edit/validation/repro nodes: rewrite or remove.

    Returns the rewritten node dict, or None if should be removed.
    """
    node_type = artifact_node['node_type']
    removed_text = '\n'.join(f'- {s}' for s in removed_fact_statements)

    # Build node content representation
    if node_type in ('issue_analysis', 'fix_plan'):
        content_field = 'text'
        content = artifact_node.get('text', '')
    elif node_type == 'code_edit':
        content = json.dumps({
            'file': artifact_node.get('file', ''),
            'description': artifact_node.get('description', ''),
            'old_str': artifact_node.get('old_str', ''),
            'new_str': artifact_node.get('new_str', ''),
        }, indent=2)
        content_field = None
    elif node_type == 'validation':
        content = json.dumps({
            'description': artifact_node.get('description', ''),
            'code': artifact_node.get('code', ''),
            'expected_output': artifact_node.get('output_after_fix', artifact_node.get('expected_output', '')),
        }, indent=2)
        content_field = None
    elif node_type == 'reproduce_script':
        content = json.dumps({
            'description': artifact_node.get('description', ''),
            'code': artifact_node.get('code', ''),
            'output_before_fix': artifact_node.get('output_before_fix', ''),
            'output_after_fix': artifact_node.get('output_after_fix', ''),
        }, indent=2)
        content_field = None
    else:
        return artifact_node  # unknown type, keep as-is

    if node_type in _REWRITABLE_TYPES:
        action_instruction = (
            "REWRITE the content to remove all parts that discuss or reference "
            "the unrelated facts below. Keep only the parts relevant to the "
            "problem statement. If after removing unrelated parts the content "
            "becomes empty or meaningless, respond with exactly: REMOVE"
        )
    else:
        action_instruction = (
            "Decide whether to REWRITE or REMOVE this artifact node.\n"
            "- If the artifact is primarily about the unrelated facts, respond: REMOVE\n"
            "- If the artifact contains parts relevant to the problem, REWRITE it "
            "to keep only the relevant parts.\n"
            "For REWRITE, output the rewritten content in JSON format matching the "
            "original structure.\n"
            "For REMOVE, respond with exactly: REMOVE"
        )

    system = (
        "You are a software engineering expert. You are editing a fact graph "
        "used to guide an AI coding agent. Some facts have been determined to "
        "be completely unrelated to the problem and removed. You now need to "
        "update artifact nodes that depended on those removed facts.\n\n"
        f"{action_instruction}\n\n"
        "When rewriting, preserve the original format and structure. "
        "Do not add new content — only remove unrelated parts."
    )
    user = (
        f"## Problem Statement\n{problem_statement}\n\n"
        f"## Removed (Unrelated) Facts\n{removed_text}\n\n"
        f"## Artifact Node (type: {node_type}, id: {artifact_node['id']})\n"
        f"### depends_on: {artifact_node.get('depends_on', [])}\n"
        f"### Content:\n{content}"
    )

    response = _llm_call(
        client, model, system, user, max_tokens=4096,
        debug_logger=debug_logger,
        debug_instance_id=debug_instance_id,
        debug_call_type=f'rewrite_{node_type}',
    )
    first_line = response.strip().split('\n')[0].strip().upper()

    if first_line == 'REMOVE' or response.strip().upper() == 'REMOVE':
        return None

    # For rewritable text nodes (analysis/plan), extract the rewritten text
    if node_type in _REWRITABLE_TYPES:
        rewritten = copy.deepcopy(artifact_node)
        # The response is the rewritten text (skip first line if it says REWRITE)
        text = response
        if text.upper().startswith('REWRITE'):
            text = '\n'.join(text.split('\n')[1:]).strip()
        # Remove markdown code fences if present
        if text.startswith('```'):
            lines = text.split('\n')
            text = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
        rewritten['text'] = text
        return rewritten

    # For JSON-based nodes (edit/validation/repro), try to parse JSON from response
    rewritten = copy.deepcopy(artifact_node)
    # Try to extract JSON from the response
    try:
        text = response
        if text.upper().startswith('REWRITE'):
            text = '\n'.join(text.split('\n')[1:]).strip()
        # Find JSON block
        if '```json' in text:
            json_start = text.index('```json') + 7
            json_end = text.index('```', json_start)
            text = text[json_start:json_end].strip()
        elif '```' in text:
            json_start = text.index('```') + 3
            json_end = text.index('```', json_start)
            text = text[json_start:json_end].strip()
        parsed = json.loads(text)
        # Update fields from parsed JSON
        for key, val in parsed.items():
            if key in rewritten and key not in ('id', 'node_type', 'depends_on'):
                rewritten[key] = val
        return rewritten
    except (json.JSONDecodeError, ValueError):
        # If we can't parse, keep the original
        return rewritten


def _ask_redundant_facts(
    client: openai.OpenAI,
    model: str,
    problem_statement: str,
    fact_nodes: list[dict],
    artifact_nodes: list[dict],
    debug_logger: DebugLogger | None = None,
    debug_instance_id: str = '',
) -> list[str]:
    """Ask LLM which fact nodes are redundant given the artifacts.

    Returns a list of fact IDs that can be safely removed.
    """
    facts_text = '\n'.join(
        f"- [{n['id']}] (depends_on: {n.get('depends_on', [])}) "
        f"{n.get('statement', '(no statement)')}"
        for n in fact_nodes
    )

    artifacts_parts: list[str] = []
    for n in artifact_nodes:
        nt = n['node_type']
        hdr = f"({n['id']}, depends_on: {n.get('depends_on', [])})"
        if nt == 'issue_analysis':
            artifacts_parts.append(f"### Analysis {hdr}:\n{n.get('text', '')}")
        elif nt == 'fix_plan':
            artifacts_parts.append(f"### Plan {hdr}:\n{n.get('text', '')}")
        elif nt == 'code_edit':
            artifacts_parts.append(
                f"### Edit {hdr}: {n.get('description', '')} "
                f"in {n.get('file', '')}\n"
                f"old_str: {n.get('old_str', '')[:200]}\n"
                f"new_str: {n.get('new_str', '')[:200]}"
            )
        elif nt == 'reproduce_script':
            artifacts_parts.append(f"### Repro {hdr}: {n.get('description', '')}")
        elif nt == 'validation':
            artifacts_parts.append(f"### Validation {hdr}: {n.get('description', '')}")
    artifacts_text = '\n\n'.join(artifacts_parts)

    system = (
        "You are a software engineering expert optimizing a fact graph that "
        "guides an AI coding agent through bug-fixing. The graph contains "
        "fact nodes (code observations) and artifact nodes (analysis, plan, "
        "edits, validation). Your task is to identify fact nodes that are "
        "REDUNDANT — the analysis, plan, and edits can be fully understood "
        "and explained without them.\n\n"
        "A fact is redundant if:\n"
        "- Its information is already captured in the analysis or plan text\n"
        "- It provides background that isn't needed to understand the edits\n"
        "- The edit is self-explanatory without this fact\n"
        "- It describes code that isn't modified and isn't needed for context\n\n"
        "A fact is NOT redundant if:\n"
        "- It explains WHY a specific edit is needed\n"
        "- It identifies the root cause described in the analysis\n"
        "- It describes the code being modified in the edits\n"
        "- It is the only dependency path from roots to an artifact node\n\n"
        "Respond with ONLY a JSON array of redundant fact IDs.\n"
        "Example: [\"f5\", \"f12\"]\nIf none are redundant: []"
    )
    user = (
        f"## Problem Statement\n{problem_statement}\n\n"
        f"## Fact Nodes\n{facts_text}\n\n"
        f"## Artifact Nodes\n{artifacts_text}"
    )
    response = _llm_call(
        client, model, system, user,
        debug_logger=debug_logger,
        debug_instance_id=debug_instance_id,
        debug_call_type='redundancy_check',
    )
    # Parse JSON array from response
    try:
        text = response.strip()
        if '```json' in text:
            text = text[text.index('```json') + 7:]
            text = text[:text.index('```')].strip()
        elif '```' in text:
            text = text[text.index('```') + 3:]
            text = text[:text.index('```')].strip()
        m = re.search(r'\[.*?\]', text, re.DOTALL)
        if m:
            text = m.group(0)
        ids = json.loads(text)
        if isinstance(ids, list):
            return [str(x) for x in ids]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


# ── Main processing ──────────────────────────────────────────────────────

def _process_instance(
    instance_id: str,
    preprocess_dir: str,
    output_dir: str,
    problem_statements: dict[str, str],
    client: openai.OpenAI,
    model: str,
    dry_run: bool = False,
    debug_logger: DebugLogger | None = None,
    phases: set[int] | None = None,
    phase1_dir: str | None = None,
) -> dict[str, Any]:
    """Process a single instance through selected reduction phases.

    Phase 1 — Irrelevance: remove non-PS-root subgraphs unrelated to the
              problem statement (LLM-assisted).
    Phase 2 — View merge: merge static facts viewing overlapping/contained
              ranges of the same file (structural, no LLM).
    Phase 3 — Redundancy removal: drop facts that aren't needed to explain
              the analysis / plan / edits (LLM-assisted).

    Args:
        phases: set of phase numbers to run (default: {1, 2, 3} = all).
        phase1_dir: if set, save phase-1 output here and reuse if it exists.

    Returns a summary dict.
    """
    if phases is None:
        phases = {1, 2, 3}
    facts_path = os.path.join(preprocess_dir, instance_id, 'stage2_facts.json')
    with open(facts_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes = data.get('nodes', [])
    ps_roots, non_ps_roots = _classify_roots(nodes)
    problem_statement = problem_statements.get(instance_id, '')

    result: dict[str, Any] = {
        'instance_id': instance_id,
        'original_nodes': len(nodes),
        'non_ps_roots': non_ps_roots,
        'subgraphs_checked': 0,
        'subgraphs_removed': 0,
        'facts_removed': 0,
        'artifacts_rewritten': 0,
        'artifacts_removed': 0,
        'view_merges': 0,
        'nodes_merged': 0,
        'facts_removed_redundant': 0,
        'final_nodes': len(nodes),
        'error': None,
    }

    working_nodes = list(nodes)

    t_instance_start = time.time()
    p1_cached = False

    # ── Phase 1: Irrelevance reduction ───────────────────────────────
    t_p1 = time.time()

    # Check if phase-1 output already exists in phase1_dir
    if phase1_dir and 1 in phases:
        p1_cache_path = os.path.join(
            phase1_dir, instance_id, 'stage2_facts.json',
        )
        if os.path.isfile(p1_cache_path):
            with open(p1_cache_path, 'r', encoding='utf-8') as f:
                p1_data = json.load(f)
            working_nodes = p1_data.get('nodes', [])
            p1_cached = True

    if 1 in phases and non_ps_roots and not p1_cached:
        if not problem_statement:
            result['error'] = 'No problem statement found'
        else:
            by_id, children = _build_maps(working_nodes)
            all_removed_fact_ids: set[str] = set()
            all_removed_fact_statements: list[str] = []

            for root_id in non_ps_roots:
                fact_subgraph = _extract_fact_subgraph(root_id, by_id, children)
                if not fact_subgraph:
                    continue
                result['subgraphs_checked'] += 1
                if dry_run:
                    continue
                try:
                    is_unrelated = _ask_relevance(
                        client, model, problem_statement, fact_subgraph,
                        debug_logger=debug_logger,
                        debug_instance_id=instance_id,
                    )
                except Exception as e:
                    result['error'] = f'LLM relevance check failed for {root_id}: {e}'
                    continue
                if is_unrelated:
                    result['subgraphs_removed'] += 1
                    removed_ids = {n['id'] for n in fact_subgraph}
                    removed_stmts = [n.get('statement', '') for n in fact_subgraph]
                    all_removed_fact_ids.update(removed_ids)
                    all_removed_fact_statements.extend(removed_stmts)
                    result['facts_removed'] += len(removed_ids)

            if all_removed_fact_ids:
                affected = _find_affected_artifacts(
                    all_removed_fact_ids, by_id, children,
                )
                artifacts_to_remove: set[str] = set()
                artifacts_rewritten: dict[str, dict] = {}

                for artifact in affected:
                    remaining_deps = [
                        d for d in artifact.get('depends_on', [])
                        if d not in all_removed_fact_ids
                    ]
                    if not remaining_deps:
                        artifacts_to_remove.add(artifact['id'])
                        result['artifacts_removed'] += 1
                        for desc_id in _get_descendants(artifact['id'], children):
                            if desc_id == artifact['id']:
                                continue
                            desc_node = by_id.get(desc_id)
                            if desc_node:
                                dr = [
                                    d for d in desc_node.get('depends_on', [])
                                    if d not in all_removed_fact_ids
                                    and d not in artifacts_to_remove
                                ]
                                if not dr:
                                    artifacts_to_remove.add(desc_id)
                                    result['artifacts_removed'] += 1
                        continue

                    if dry_run:
                        continue
                    try:
                        rw = _ask_rewrite_artifact(
                            client, model, problem_statement, artifact,
                            all_removed_fact_statements,
                            debug_logger=debug_logger,
                            debug_instance_id=instance_id,
                        )
                        if rw is None:
                            artifacts_to_remove.add(artifact['id'])
                            result['artifacts_removed'] += 1
                        else:
                            rw['depends_on'] = remaining_deps
                            artifacts_rewritten[artifact['id']] = rw
                            result['artifacts_rewritten'] += 1
                    except Exception as e:
                        result['error'] = (
                            f'LLM rewrite failed for {artifact["id"]}: {e}'
                        )
                        upd = copy.deepcopy(artifact)
                        upd['depends_on'] = remaining_deps
                        artifacts_rewritten[artifact['id']] = upd

                # Cascade removal
                changed = True
                while changed:
                    changed = False
                    for n in list(by_id.values()):
                        if (n['id'] in all_removed_fact_ids
                                or n['id'] in artifacts_to_remove):
                            continue
                        deps = n.get('depends_on', [])
                        if not deps:
                            continue
                        rem = [
                            d for d in deps
                            if d not in all_removed_fact_ids
                            and d not in artifacts_to_remove
                        ]
                        if not rem:
                            artifacts_to_remove.add(n['id'])
                            result['artifacts_removed'] += 1
                            changed = True

                # Build phase-1 output
                all_removed = all_removed_fact_ids | artifacts_to_remove
                p1_nodes: list[dict] = []
                for node in working_nodes:
                    nid = node['id']
                    if nid in all_removed:
                        continue
                    if nid in artifacts_rewritten:
                        p1_nodes.append(artifacts_rewritten[nid])
                    else:
                        upd = copy.deepcopy(node)
                        upd['depends_on'] = [
                            d for d in upd.get('depends_on', [])
                            if d not in all_removed
                        ]
                        p1_nodes.append(upd)
                working_nodes = p1_nodes
    result['nodes_after_p1'] = len(working_nodes)
    result['time_phase1_s'] = round(time.time() - t_p1, 2)
    result['phase1_cached'] = p1_cached

    # Save phase-1 output if phase1_dir is set (and we actually ran phase 1)
    if phase1_dir and 1 in phases and not p1_cached:
        p1_out = copy.deepcopy(data)
        p1_out['nodes'] = working_nodes
        _save_graph(p1_out, instance_id, phase1_dir)

    # ── Phase 2: Merge overlapping view facts (structural, no LLM) ──
    t_p2 = time.time()
    if 2 in phases:
        working_nodes, merge_stats = _merge_view_nodes(working_nodes)
        result['view_merges'] = merge_stats['merges']
        result['nodes_merged'] = merge_stats['nodes_merged']
    result['nodes_after_p2'] = len(working_nodes)
    result['time_phase2_s'] = round(time.time() - t_p2, 2)

    # ── Phase 3: Remove redundant facts (LLM-assisted) ──────────────
    t_p3 = time.time()
    if 3 in phases and not dry_run and problem_statement:
        fact_nodes = [
            n for n in working_nodes if n.get('node_type') == _FACT
        ]
        artifact_nodes = [
            n for n in working_nodes if n.get('node_type') in _ARTIFACT_TYPES
        ]
        if fact_nodes and artifact_nodes:
            try:
                redundant_ids = _ask_redundant_facts(
                    client, model, problem_statement,
                    fact_nodes, artifact_nodes,
                    debug_logger=debug_logger,
                    debug_instance_id=instance_id,
                )
                valid_fact_ids = {n['id'] for n in fact_nodes}
                redundant_ids = [
                    rid for rid in redundant_ids if rid in valid_fact_ids
                ]
                if redundant_ids:
                    working_nodes = _remove_facts_and_update_deps(
                        working_nodes, set(redundant_ids),
                    )
                    result['facts_removed_redundant'] = len(redundant_ids)
            except Exception as e:
                err = f'Redundancy check failed: {e}'
                result['error'] = (
                    f"{result['error']}; {err}" if result['error'] else err
                )
    result['time_phase3_s'] = round(time.time() - t_p3, 2)
    result['nodes_after_p3'] = len(working_nodes)

    # ── Save final graph ─────────────────────────────────────────────
    new_data = copy.deepcopy(data)
    new_data['nodes'] = working_nodes
    new_data['_reduction_meta'] = {
        'original_node_count': len(nodes),
        'reduced_node_count': len(working_nodes),
        'view_merges': result['view_merges'],
        'nodes_merged': result['nodes_merged'],
        'facts_removed_redundant': result['facts_removed_redundant'],
    }

    result['final_nodes'] = len(working_nodes)
    result['time_total_s'] = round(time.time() - t_instance_start, 2)
    _save_graph(new_data, instance_id, output_dir)
    return result


def _save_graph(data: dict, instance_id: str, output_dir: str) -> None:
    out_path = os.path.join(output_dir, instance_id, 'stage2_facts.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='LLM-assisted fact graph reduction')
    parser.add_argument('--preprocess-dir', required=True, help='Input swegym_v6 directory')
    parser.add_argument('--filter-json', required=True, help='Filter JSON from filter_fact_graphs.py')
    parser.add_argument('--output-dir', required=True, help='Output directory for reduced graphs')
    parser.add_argument('--api-base', default='http://localhost:8000/v1', help='LLM API base URL')
    parser.add_argument('--model', default='glm-5', help='Model name for API')
    parser.add_argument('--api-key', default='EMPTY', help='API key')
    parser.add_argument('--dataset', default='SWE-Gym/SWE-Gym', help='HuggingFace dataset')
    parser.add_argument('--split', default='train', help='Dataset split')
    parser.add_argument('--max-workers', type=int, default=4, help='Parallel workers')
    parser.add_argument('--dry-run', action='store_true', help='Analyze without calling LLM')
    parser.add_argument('--phases', type=str, default='1,2,3',
                        help='Comma-separated list of phases to run: 1=irrelevance, 2=view-merge, 3=redundancy (default: 1,2,3)')
    parser.add_argument('--phase1-dir', default=None,
                        help='Directory to save/cache phase-1 output. If set and phase-1 output exists, skip phase 1 and reuse cached result.')
    parser.add_argument('--debug-log-dir', default=None, help='Directory to save LLM prompt/response debug logs (JSONL per instance)')
    parser.add_argument('--instance-ids', nargs='+', default=None, help='Process specific instances')
    args = parser.parse_args()

    # Parse phases
    active_phases = set(int(x.strip()) for x in args.phases.split(',') if x.strip())
    if not active_phases.issubset({1, 2, 3}):
        parser.error(f'Invalid phases: {active_phases}. Must be subset of {{1, 2, 3}}')

    # Load filter results
    with open(args.filter_json, 'r', encoding='utf-8') as f:
        filter_data = json.load(f)
    filtered_ids = set(filter_data.get('filtered_instance_ids', []))
    remaining_ids = set(filter_data.get('remaining_instance_ids', []))

    print(f'Filter: {len(filtered_ids)} filtered, {len(remaining_ids)} remaining')

    # Determine which instances to process
    if args.instance_ids:
        target_ids = [iid for iid in args.instance_ids if iid in remaining_ids]
    else:
        target_ids = sorted(remaining_ids)

    # Classify instances by whether they have non-PS roots
    # (Phase 1 only applies to those with non-PS roots, but Phase 2/3 apply to all)
    instances_with_non_ps: list[str] = []
    instances_no_non_ps: list[str] = []
    all_valid_instances: list[str] = []
    for iid in target_ids:
        facts_path = os.path.join(args.preprocess_dir, iid, 'stage2_facts.json')
        if not os.path.isfile(facts_path):
            continue
        all_valid_instances.append(iid)
        with open(facts_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _, non_ps = _classify_roots(data.get('nodes', []))
        if non_ps:
            instances_with_non_ps.append(iid)
        else:
            instances_no_non_ps.append(iid)

    print(f'Instances with non-PS roots: {len(instances_with_non_ps)}')
    print(f'Instances without non-PS roots: {len(instances_no_non_ps)}')
    print(f'Total to process: {len(all_valid_instances)}')

    # Load problem statements from HuggingFace dataset
    print(f'Loading dataset: {args.dataset} ({args.split})...')
    ds = load_dataset(args.dataset, split=args.split)
    problem_statements = {row['instance_id']: row['problem_statement'] for row in ds}
    print(f'Loaded {len(problem_statements)} problem statements')

    # Initialize LLM client and debug logger
    client = openai.OpenAI(base_url=args.api_base, api_key=args.api_key)
    debug_logger = DebugLogger(args.debug_log_dir)
    if debug_logger.enabled:
        print(f'Debug logging enabled: {args.debug_log_dir}')
    print(f'Active phases: {sorted(active_phases)}')
    if args.phase1_dir:
        print(f'Phase-1 cache dir: {args.phase1_dir}')

    # Process ALL instances (Phase 1 self-gates on non_ps_roots, Phase 2/3 apply to all)
    results: list[dict] = []

    def _worker(iid: str) -> dict:
        try:
            return _process_instance(
                iid, args.preprocess_dir, args.output_dir,
                problem_statements, client, args.model, args.dry_run,
                debug_logger=debug_logger,
                phases=active_phases,
                phase1_dir=args.phase1_dir,
            )
        except Exception as e:
            traceback.print_exc()
            return {'instance_id': iid, 'error': str(e)}

    if args.max_workers <= 1 or args.dry_run:
        for i, iid in enumerate(all_valid_instances):
            print(f'[{i + 1}/{len(all_valid_instances)}] Processing {iid}...')
            r = _worker(iid)
            results.append(r)
            print(f'  {r.get("original_nodes","?")}'
                  f' -P1{"(cached)" if r.get("phase1_cached") else ""}-> {r.get("nodes_after_p1","?")}'
                  f' -P2-> {r.get("nodes_after_p2","?")}'
                  f' -P3-> {r.get("nodes_after_p3","?")}'
                  f'  ({r.get("time_phase1_s",0)}s {r.get("time_phase2_s",0)}s {r.get("time_phase3_s",0)}s'
                  f' total={r.get("time_total_s",0)}s)'
                  f'{"  ERROR: "+r["error"] if r.get("error") else ""}')
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(_worker, iid): iid for iid in all_valid_instances}
            done = 0
            for future in as_completed(futures):
                done += 1
                iid = futures[future]
                r = future.result()
                results.append(r)
                print(f'[{done}/{len(all_valid_instances)}] {iid}: '
                      f'{r.get("original_nodes","?")}'
                      f' -P1{"(cached)" if r.get("phase1_cached") else ""}-> {r.get("nodes_after_p1","?")}'
                      f' -P2-> {r.get("nodes_after_p2","?")}'
                      f' -P3-> {r.get("nodes_after_p3","?")}'
                      f'  ({r.get("time_phase1_s",0)}s {r.get("time_phase2_s",0)}s {r.get("time_phase3_s",0)}s'
                      f' total={r.get("time_total_s",0)}s)'
                      f'{"  ERROR: "+r["error"] if r.get("error") else ""}')

    # Write reduction summary
    summary = {
        'preprocess_dir': args.preprocess_dir,
        'output_dir': args.output_dir,
        'model': args.model,
        'dry_run': args.dry_run,
        'total_processed': len(results),
        'instances_with_non_ps_roots': len(instances_with_non_ps),
        'instances_without_non_ps_roots': len(instances_no_non_ps),
        'subgraphs_checked': sum(r.get('subgraphs_checked', 0) for r in results),
        'subgraphs_removed': sum(r.get('subgraphs_removed', 0) for r in results),
        'total_facts_removed': sum(r.get('facts_removed', 0) for r in results),
        'total_artifacts_removed': sum(r.get('artifacts_removed', 0) for r in results),
        'total_artifacts_rewritten': sum(r.get('artifacts_rewritten', 0) for r in results),
        'total_view_merges': sum(r.get('view_merges', 0) for r in results),
        'total_nodes_merged': sum(r.get('nodes_merged', 0) for r in results),
        'total_facts_removed_redundant': sum(r.get('facts_removed_redundant', 0) for r in results),
        'errors': [r for r in results if r.get('error')],
        'per_instance': results,
    }

    summary_path = os.path.join(args.output_dir, '_reduction_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'\n=== Reduction Summary ===')
    print(f'Processed: {len(results)}, Copied: {len(instances_no_non_ps)}')
    print(f'Phase 1 — Irrelevance: subgraphs checked={summary["subgraphs_checked"]}, '
          f'removed={summary["subgraphs_removed"]}, '
          f'facts={summary["total_facts_removed"]}, '
          f'artifacts removed={summary["total_artifacts_removed"]}, '
          f'rewritten={summary["total_artifacts_rewritten"]}')
    print(f'Phase 2 — View merge: merges={summary["total_view_merges"]}, '
          f'nodes merged={summary["total_nodes_merged"]}')
    print(f'Phase 3 — Redundancy: facts removed={summary["total_facts_removed_redundant"]}')
    if summary['errors']:
        print(f'Errors: {len(summary["errors"])}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
