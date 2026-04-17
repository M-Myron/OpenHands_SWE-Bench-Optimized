"""Fact tracker V2 for investigation graphs (stageless).

Parses ``stage2_facts.json`` files with node types: ``fact``,
``reproduce_script``, ``issue_analysis``, ``fix_plan``, ``code_edit``,
``validation``.  Maintains a boolean usage map and DAG-based availability.

Unlike V1, this tracker has NO stage logic — facts and artifacts are all
treated uniformly through DAG availability.  Stage-based gating is replaced
by the SufficiencyCritic in the agent orchestration layer.
"""

from __future__ import annotations

import re
from typing import Any

from openhands.core.logger import openhands_logger as logger


# Node-type constants
_FACT = 'fact'
_REPRO = 'reproduce_script'
_ANALYSIS = 'issue_analysis'
_PLAN = 'fix_plan'
_EDIT = 'code_edit'
_VALIDATION = 'validation'

_ARTIFACT_TYPES = {_REPRO, _ANALYSIS, _PLAN, _EDIT, _VALIDATION}


class FactTrackerV2:
    """Manages investigation facts with DAG-gated availability (stageless)."""

    def __init__(self, data: dict) -> None:
        self._nodes: dict[str, dict] = {}
        self._used: dict[str, bool] = {}
        self._used_at_step: dict[str, int] = {}  # node_id -> step when marked used
        self._children: dict[str, list[str]] = {}  # parent -> children
        self._unlocker_satisfied: set[str] = set()  # fact IDs whose unlocker is met
        self._all_impl_done_at_step: int | None = None
        self._finish_extension_budget: int = 10

        nodes = data.get('nodes', [])
        for node in nodes:
            nid = node['id']
            self._nodes[nid] = node
            self._used[nid] = False

        # Build child map for debugging/stats
        for nid, node in self._nodes.items():
            for dep in node.get('depends_on', []):
                self._children.setdefault(dep, []).append(nid)

        # Classify roots
        self._ps_roots: list[str] = []
        self._non_ps_roots: list[str] = []
        for nid, node in self._nodes.items():
            deps = node.get('depends_on', [])
            if not deps:
                unlocker = node.get('unlocker', {})
                action = unlocker.get('action', '') if isinstance(unlocker, dict) else ''
                if '[view] problem_statement' in action.lower() or 'problem_statement' in action.lower():
                    self._ps_roots.append(nid)
                else:
                    self._non_ps_roots.append(nid)

        logger.info(
            f'[FactTrackerV2] Loaded {len(self._nodes)} nodes: '
            f'{len(self._ps_roots)} PS-roots, {len(self._non_ps_roots)} non-PS-roots'
        )

        # Auto-mark problem_statement roots as used — the issue text is
        # always visible in the prompt so these are satisfied from step 0.
        for nid in self._ps_roots:
            self._used[nid] = True
            self._used_at_step[nid] = 0
        if self._ps_roots:
            logger.info(
                f'[FactTrackerV2] Auto-marked {len(self._ps_roots)} '
                f'problem_statement roots as used: {self._ps_roots}'
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_facts(self) -> bool:
        return len(self._nodes) > 0

    @property
    def node_ids(self) -> list[str]:
        return list(self._nodes.keys())

    def _is_fact_node(self, node: dict) -> bool:
        """Return True if this node is a fact-level node (not an artifact)."""
        return node.get('node_type', '') not in _ARTIFACT_TYPES

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def _preconditions_met(self, node_id: str) -> bool:
        """Check if all depends_on nodes are used."""
        deps = self._nodes[node_id].get('depends_on', [])
        for d in deps:
            if not self._used.get(d, False):
                return False
        return True

    def get_available_nodes(self) -> list[dict]:
        """Return nodes that are not used and whose preconditions are all used."""
        result = []
        for nid, node in self._nodes.items():
            if self._used[nid]:
                continue
            if self._preconditions_met(nid):
                result.append(node)
        return result

    def get_available_node_ids(self) -> list[str]:
        return [n['id'] for n in self.get_available_nodes()]

    # ------------------------------------------------------------------
    # Usage
    # ------------------------------------------------------------------

    def mark_used(self, node_ids: list[str], step_index: int = -1, force: bool = False) -> None:
        """Mark nodes as used. DAG preconditions are always enforced.

        Exception: nodes whose unlocker is already satisfied (evidence
        visible in history) can bypass preconditions.

        If ``force=True``, DAG preconditions are skipped entirely.
        Use this when an external validator (e.g. SufficiencyCritic)
        has independently confirmed the artifact is equivalent.
        """
        for nid in node_ids:
            # Resolve: if nid is unknown, try to match it as a node_type
            if nid not in self._used:
                resolved = self._resolve_node_id(nid)
                if resolved:
                    logger.info(
                        f'[FactTrackerV2] Resolved "{nid}" → "{resolved}" '
                        f'(matched by node_type).'
                    )
                    nid = resolved
                else:
                    logger.warning(f'[FactTrackerV2] Unknown node ID: {nid}')
                    continue
            if self._used[nid]:
                continue  # already used, skip silently

            # Force bypass — skip all precondition checks
            if force:
                self._used[nid] = True
                self._used_at_step[nid] = step_index
                logger.info(
                    f'[FactTrackerV2] Marked {nid} as used '
                    f'(step {step_index}, force bypass).'
                )
                continue

            # Facts whose unlocker is already satisfied can bypass preconditions
            if nid in self._unlocker_satisfied:
                self._used[nid] = True
                self._used_at_step[nid] = step_index
                logger.info(
                    f'[FactTrackerV2] Marked {nid} as used '
                    f'(step {step_index}, unlocker-satisfied bypass).'
                )
                continue

            # Enforce DAG preconditions
            if not self._preconditions_met(nid):
                unmet = [
                    d for d in self._nodes[nid].get('depends_on', [])
                    if not self._used.get(d, False)
                ]
                logger.warning(
                    f'[FactTrackerV2] Cannot mark {nid} as used: '
                    f'unmet preconditions {unmet}. Skipping.'
                )
                continue
            self._used[nid] = True
            self._used_at_step[nid] = step_index
            logger.info(f'[FactTrackerV2] Marked {nid} as used (step {step_index}).')

    def _resolve_node_id(self, candidate: str) -> str | None:
        """Try to resolve an unknown ID by matching against node_type."""
        for nid, node in self._nodes.items():
            if node.get('node_type') == candidate:
                return nid
        for nid, node in self._nodes.items():
            ntype = node.get('node_type', '')
            if ntype and (candidate.startswith(ntype) or ntype.startswith(candidate)):
                return nid
        return None

    def is_used(self, node_id: str) -> bool:
        return self._used.get(node_id, False)

    # ------------------------------------------------------------------
    # Queries for SufficiencyCritic
    # ------------------------------------------------------------------

    def get_unused_facts_and_artifacts(self) -> dict[str, list[dict]]:
        """Return unused facts and unused artifacts separately.

        Returns dict with keys:
        - ``unused_facts``: list of unused fact-type nodes (id, statement)
        - ``unused_artifacts``: list of unused artifact nodes (id, node_type, description)
        """
        unused_facts: list[dict] = []
        unused_artifacts: list[dict] = []

        for nid, node in self._nodes.items():
            if self._used.get(nid, False):
                continue
            if self._is_fact_node(node):
                unused_facts.append({
                    'id': nid,
                    'statement': node.get('statement', ''),
                    'type': node.get('type', 'unknown'),
                })
            else:
                ntype = node.get('node_type', '')
                art_entry: dict[str, Any] = {
                    'id': nid,
                    'node_type': ntype,
                    'description': node.get('description', node.get('text', ntype)),
                    'depends_on': node.get('depends_on', []),
                }
                # Include artifact content for equivalence checking
                if ntype == _REPRO and node.get('code'):
                    art_entry['code'] = node['code']
                elif ntype in (_ANALYSIS, _PLAN) and node.get('text'):
                    art_entry['text'] = node['text']
                unused_artifacts.append(art_entry)

        return {
            'unused_facts': unused_facts,
            'unused_artifacts': unused_artifacts,
        }

    def get_unused_deps_for_artifact_type(self, artifact_type: str) -> list[dict]:
        """Return unused fact nodes that a given artifact type depends on.

        Walks the ``depends_on`` list of every unused artifact of
        *artifact_type* and collects the fact-level dependencies that
        are themselves still unused.  This tells the planner exactly
        which facts must be explored before the artifact can be unlocked.
        """
        dep_ids: set[str] = set()
        for nid, node in self._nodes.items():
            if node.get('node_type') != artifact_type:
                continue
            if self._used.get(nid, False):
                continue
            for d in node.get('depends_on', []):
                if not self._used.get(d, False):
                    dep_ids.add(d)

        result: list[dict] = []
        for did in sorted(dep_ids):
            dnode = self._nodes.get(did)
            if dnode and self._is_fact_node(dnode):
                result.append({
                    'id': did,
                    'statement': dnode.get('statement', ''),
                })
        return result

    def get_used_facts_summary(self) -> list[dict]:
        """Return used facts as a structured list for critic input."""
        result: list[dict] = []
        for nid, node in self._nodes.items():
            if not self._used.get(nid, False):
                continue
            if self._is_fact_node(node):
                result.append({
                    'id': nid,
                    'statement': node.get('statement', ''),
                    'step': self._used_at_step.get(nid, -1),
                })
            else:
                result.append({
                    'id': nid,
                    'node_type': node.get('node_type', ''),
                    'description': node.get('description', node.get('text', '')),
                    'step': self._used_at_step.get(nid, -1),
                })
        return result

    def is_artifact_used(self, artifact_type: str) -> bool:
        """Check if any artifact of the given type has been marked as used."""
        for nid, node in self._nodes.items():
            if node.get('node_type') == artifact_type and self._used.get(nid, False):
                return True
        return False

    # ------------------------------------------------------------------
    # Phase gate checking (kept for compatibility / queries)
    # ------------------------------------------------------------------

    def get_nodes_by_type(self, node_type: str) -> list[dict]:
        """Return all nodes of a given type."""
        return [n for n in self._nodes.values() if n.get('node_type') == node_type]

    def get_phase_progress(self) -> dict[str, Any]:
        """Return which artifact phases are complete."""
        def _any_used(ntype: str) -> bool:
            return any(
                self._used[nid]
                for nid, n in self._nodes.items()
                if n.get('node_type') == ntype
            )

        def _all_used(ntype: str) -> bool:
            ids = [nid for nid, n in self._nodes.items() if n.get('node_type') == ntype]
            return bool(ids) and all(self._used[nid] for nid in ids)

        def _used_ids(ntype: str) -> list[str]:
            return [
                nid for nid, n in self._nodes.items()
                if n.get('node_type') == ntype and self._used[nid]
            ]

        return {
            'repro_done': _any_used(_REPRO),
            'analysis_done': _any_used(_ANALYSIS),
            'plan_done': _any_used(_PLAN),
            'edits_used': _used_ids(_EDIT),
            'all_edits_done': _all_used(_EDIT),
            'validations_used': _used_ids(_VALIDATION),
            'all_validations_done': _all_used(_VALIDATION),
        }

    def get_usage_summary(self) -> dict[str, Any]:
        total = len(self._nodes)
        used = sum(1 for v in self._used.values() if v)
        available = len(self.get_available_nodes())
        blocked = total - used - available
        fact_nodes = sum(1 for n in self._nodes.values() if self._is_fact_node(n))
        fact_used = sum(
            1 for nid, n in self._nodes.items()
            if self._is_fact_node(n) and self._used[nid]
        )
        artifact_nodes = sum(1 for n in self._nodes.values() if n.get('node_type') in _ARTIFACT_TYPES)
        artifact_used = sum(
            1 for nid, n in self._nodes.items()
            if n.get('node_type') in _ARTIFACT_TYPES and self._used[nid]
        )
        return {
            'total_nodes': total,
            'used_nodes': used,
            'available_nodes': available,
            'blocked_nodes': blocked,
            'fact_nodes': fact_nodes,
            'fact_used': fact_used,
            'artifact_nodes': artifact_nodes,
            'artifact_used': artifact_used,
            'used_ids': [nid for nid, v in self._used.items() if v],
        }

    # ------------------------------------------------------------------
    # Rendering for planner
    # ------------------------------------------------------------------

    def render_used_facts_summary(self) -> str:
        """Render a compact summary of already-used facts for planner context."""
        used_nodes = [
            (nid, self._nodes[nid])
            for nid in self._nodes
            if self._used.get(nid, False)
        ]
        if not used_nodes:
            return ''

        lines: list[str] = []
        for nid, node in used_nodes:
            step = self._used_at_step.get(nid, -1)
            ntype = node.get('node_type', 'fact')
            if self._is_fact_node(node):
                summary = node.get('statement', '')
            else:
                summary = node.get('description', node.get('text', ntype))
            if len(summary) > 200:
                summary = summary[:197] + '...'
            lines.append(f'- **[{nid}]** (step {step}, {ntype}): {summary}')

        return '\n'.join(lines)

    def render_available_facts_for_planner(self) -> str:
        """Render all available facts grouped by type for the planner prompt."""
        available = self.get_available_nodes()
        if not available:
            return '(No facts currently available — all are either used or blocked by unmet preconditions.)'

        ps_root_facts = []
        non_ps_root_facts = []
        regular_facts = []
        artifacts = []

        for node in available:
            nid = node['id']
            if nid in self._ps_roots:
                ps_root_facts.append(node)
            elif nid in self._non_ps_roots:
                non_ps_root_facts.append(node)
            elif self._is_fact_node(node):
                regular_facts.append(node)
            else:
                artifacts.append(node)

        parts: list[str] = []

        if ps_root_facts:
            parts.append('### Problem-Statement Root Facts')
            parts.append('These facts are derived from the problem statement and are immediately available.\n')
            for n in ps_root_facts:
                parts.append(self._render_fact_node(n))

        if non_ps_root_facts:
            parts.append('### Non-Problem-Statement Root Facts')
            parts.append(
                'These facts have no DAG preconditions but their unlockers reference code/files '
                'NOT mentioned in the problem statement. You can use them when:\n'
                '  (a) The solver has already discovered the relevant file/code in history, OR\n'
                '  (b) You guide the solver to naturally explore that code area with a '
                'logically connected rationale from the investigation.\n'
                'Do NOT randomly suggest viewing an unrelated file.\n'
            )
            for n in non_ps_root_facts:
                parts.append(self._render_fact_node(n))

        if regular_facts:
            parts.append('### Available Investigation Facts')
            parts.append('These facts have all preconditions met and are ready to be unlocked.\n')
            for n in regular_facts:
                parts.append(self._render_fact_node(n))

        if artifacts:
            parts.append('### Available Artifact Nodes')
            parts.append('These artifacts have all preconditions met and are ready to be produced.\n')
            for n in artifacts:
                parts.append(self._render_artifact_node(n))

        return '\n'.join(parts)

    def render_categorized_facts(self, history_text: str) -> str:
        """Render facts categorized by unlocker satisfaction status.

        Categories:
        - Unlocker Satisfied — Needs Articulation
        - Available — Unlocker Not Yet Satisfied (static)
        - Available Dynamic — Unlocker Requires Execution
        - Available Artifacts
        """
        cats = self.categorize_available_facts(history_text)

        parts: list[str] = []

        if cats['unlocked_not_articulated']:
            parts.append('### Unlocker Satisfied — Needs Articulation')
            parts.append(
                'The unlocker action for these facts has been performed — '
                'the relevant code/output is visible in history. The solver has '
                'not yet articulated the finding. Revise the candidate to add '
                'reasoning that interprets what was discovered, then mark as used.\n'
            )
            for n in cats['unlocked_not_articulated']:
                parts.append(self._render_fact_node(n))

        if cats['available_not_unlocked_static']:
            parts.append('### Available — Unlocker Not Yet Satisfied')
            parts.append(
                'Ready to target but the unlocker action has not been performed. '
                'Choose your next target from here.\n'
            )
            # Separate non-PS root facts
            non_ps = [n for n in cats['available_not_unlocked_static'] if n['id'] in self._non_ps_roots]
            regular = [n for n in cats['available_not_unlocked_static'] if n['id'] not in self._non_ps_roots]

            if non_ps:
                parts.append('#### Non-Problem-Statement Root Facts')
                parts.append(
                    'These reference code NOT mentioned in the problem statement. '
                    'Use only when naturally connected to prior investigation.\n'
                )
                for n in non_ps:
                    parts.append(self._render_fact_node(n))

            if regular:
                if non_ps:
                    # Add sub-header to visually separate from non-PS roots
                    parts.append('#### Available Investigation Facts')
                    parts.append(
                        'These facts have all preconditions met and are ready to be unlocked.\n'
                    )
                for n in regular:
                    parts.append(self._render_fact_node(n))

        if cats['available_not_unlocked_dynamic']:
            parts.append('### Available Dynamic — Unlocker Requires Execution')
            parts.append(
                'These have a [bash] unlocker — include reasoning + execution in one step.\n'
            )
            for n in cats['available_not_unlocked_dynamic']:
                parts.append(self._render_fact_node(n))

        if cats['artifacts']:
            parts.append('### Available Artifact Nodes')
            parts.append('These artifacts have all preconditions met and are ready to be produced.\n')
            for n in cats['artifacts']:
                parts.append(self._render_artifact_node(n))

        if not parts:
            return '(No facts currently available — all are either used or blocked by unmet preconditions.)'

        return '\n'.join(parts)

    def _render_fact_node(self, node: dict) -> str:
        nid = node['id']
        ntype = node.get('type', 'unknown')
        statement = node.get('statement', '')
        unlocker = node.get('unlocker', {})
        action = unlocker.get('action', '') if isinstance(unlocker, dict) else ''
        observation = unlocker.get('observation', '') if isinstance(unlocker, dict) else ''
        deps = node.get('depends_on', [])
        deps_str = ', '.join(deps) if deps else 'none'

        lines = [
            f'**[{nid}]** (type: {ntype}, depends_on: {deps_str})',
            f'  Statement: {statement}',
            f'  Unlocker action: {action}',
            f'  Unlocker observation: {observation}',
            '',
        ]
        return '\n'.join(lines)

    def _render_artifact_node(self, node: dict) -> str:
        nid = node['id']
        ntype = node.get('node_type', '')
        deps = node.get('depends_on', [])
        deps_str = ', '.join(deps) if deps else 'none'
        lines = [f'**[{nid}]** (type: {ntype}, depends_on: {deps_str})']

        if ntype == _REPRO:
            lines.append(f'  Description: {node.get("description", "")}')
            code = node.get('code', '')
            if code:
                lines.append(f'  Code:\n```python\n{code}\n```')
            lines.append(f'  Output before fix: {node.get("output_before_fix", "")}')
            lines.append(f'  Output after fix: {node.get("output_after_fix", "")}')
        elif ntype == _ANALYSIS:
            lines.append(f'  Analysis text:\n{node.get("text", "")}')
        elif ntype == _PLAN:
            lines.append(f'  Fix plan:\n{node.get("text", "")}')
        elif ntype == _EDIT:
            lines.append(f'  Description: {node.get("description", "")}')
            lines.append(f'  File: {node.get("file", "")}')
            lines.append(f'  Action type: {node.get("action_type", "")}')
            old_str = node.get('old_str', '')
            new_str = node.get('new_str', '')
            if old_str:
                lines.append(f'  Old code:\n```\n{old_str}\n```')
            if new_str:
                lines.append(f'  New code:\n```\n{new_str}\n```')
        elif ntype == _VALIDATION:
            lines.append(f'  Description: {node.get("description", "")}')
            lines.append(f'  Command: {node.get("command", "")}')
            lines.append(f'  Expected output: {node.get("expected_output", "")}')

        lines.append('')
        return '\n'.join(lines)

    def render_usage_state_for_planner(self) -> str:
        """Render a concise summary of fact usage for the planner."""
        summary = self.get_usage_summary()
        progress = self.get_phase_progress()

        lines = [
            '### Fact Usage State',
            f'Total nodes: {summary["total_nodes"]} | '
            f'Used: {summary["used_nodes"]} | '
            f'Available: {summary["available_nodes"]} | '
            f'Blocked: {summary["blocked_nodes"]}',
            f'Facts: {summary["fact_used"]}/{summary["fact_nodes"]} used | '
            f'Artifacts: {summary["artifact_used"]}/{summary["artifact_nodes"]} used',
            '',
            '### Phase Progress',
            f'  Reproduce script done: {"YES" if progress["repro_done"] else "NO"}',
            f'  Issue analysis done:   {"YES" if progress["analysis_done"] else "NO"}',
            f'  Fix plan done:         {"YES" if progress["plan_done"] else "NO"}',
            f'  All edits done:        {"YES" if progress["all_edits_done"] else "NO"}'
            + (f' (completed: {", ".join(progress["edits_used"])})' if progress["edits_used"] else ''),
            f'  All validations done:  {"YES" if progress["all_validations_done"] else "NO"}'
            + (f' (completed: {", ".join(progress["validations_used"])})' if progress["validations_used"] else ''),
        ]

        if summary['used_nodes'] > 0:
            lines.append(f'\nUsed node IDs: {", ".join(summary["used_ids"])}')

        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Finish tracking
    # ------------------------------------------------------------------

    def all_facts_used(self) -> bool:
        """Return True if all fact-type nodes are used."""
        for nid, node in self._nodes.items():
            if self._is_fact_node(node) and not self._used[nid]:
                return False
        return True

    def all_nodes_used(self) -> bool:
        """Return True if all nodes (facts + artifacts) are used."""
        return all(self._used.values())

    def set_finish_extension_budget(self, budget: int) -> None:
        self._finish_extension_budget = budget

    def check_and_set_impl_complete(self, current_step: int) -> bool:
        """Check if all edits+validations are done; record the step if so."""
        progress = self.get_phase_progress()
        if not (progress['all_edits_done'] and progress['all_validations_done']):
            return False
        if self._all_impl_done_at_step is not None:
            return False
        self._all_impl_done_at_step = current_step
        logger.info(
            f'[FactTrackerV2] All edits+validations done at step '
            f'{current_step}. Finish extension budget: '
            f'{self._finish_extension_budget} steps.'
        )
        return True

    def should_force_finish(self, current_step: int) -> bool:
        if self._all_impl_done_at_step is None:
            return False
        steps_since = current_step - self._all_impl_done_at_step
        return steps_since >= self._finish_extension_budget

    def get_finish_countdown(self, current_step: int) -> int | None:
        if self._all_impl_done_at_step is None:
            return None
        remaining = self._finish_extension_budget - (current_step - self._all_impl_done_at_step)
        return max(0, remaining)

    # ------------------------------------------------------------------
    # Unlocker satisfaction checking
    # ------------------------------------------------------------------

    _VIEW_PATTERN = re.compile(
        r'\[view\]\s+(\S+?)(?:\s+(\d+)(?:-(\d+))?)?$'
    )

    @classmethod
    def parse_static_unlocker(cls, action: str) -> dict | None:
        """Parse a static unlocker action string."""
        action = action.strip()
        if not action:
            return None

        if 'problem_statement' in action.lower():
            return {'type': 'problem_statement'}

        m = cls._VIEW_PATTERN.match(action)
        if m:
            filepath = m.group(1)
            start = int(m.group(2)) if m.group(2) else None
            end = int(m.group(3)) if m.group(3) else (start if start else None)
            return {
                'type': 'view',
                'file_path': filepath,
                'start_line': start,
                'end_line': end,
            }

        if action.startswith('[bash]'):
            return {'type': 'bash'}

        return None

    @staticmethod
    def check_view_unlocker_in_history(
        file_path: str,
        start_line: int | None,
        end_line: int | None,
        history_text: str,
    ) -> bool:
        """Check whether the interaction history contains the content
        from the specified file at the specified line range."""
        if not history_text or not file_path:
            return False

        escaped_path = re.escape(file_path)
        path_pattern = rf'(?:^|[\s/])(?:\S*/)?{escaped_path}'
        if not re.search(path_pattern, history_text, re.MULTILINE):
            return False

        if start_line is None:
            return True

        path_match = re.search(path_pattern, history_text, re.MULTILINE)
        if not path_match:
            return False

        remaining = history_text[path_match.start():]

        for line_num in range(start_line, (end_line or start_line) + 1):
            line_pattern = rf'(?:^|\s){line_num}[\t\s:|]'
            if re.search(line_pattern, remaining):
                return True

        return False

    def check_fact_unlocker_satisfied(
        self, node_id: str, history_text: str,
    ) -> bool:
        """Check if a fact's static unlocker is satisfied by the history."""
        node = self._nodes.get(node_id)
        if not node or not self._is_fact_node(node):
            return False

        unlocker = node.get('unlocker', {})
        action = unlocker.get('action', '') if isinstance(unlocker, dict) else ''
        parsed = self.parse_static_unlocker(action)

        if not parsed:
            return False

        if parsed['type'] == 'problem_statement':
            return True

        if parsed['type'] == 'view':
            return self.check_view_unlocker_in_history(
                parsed['file_path'],
                parsed['start_line'],
                parsed['end_line'],
                history_text,
            )

        return False

    def categorize_available_facts(
        self, history_text: str,
    ) -> dict[str, list[dict]]:
        """Categorize available fact nodes based on unlocker satisfaction.

        Returns a dict with keys:
        - 'unlocked_not_articulated': facts whose evidence IS visible in
          history but not yet marked as used.
        - 'available_not_unlocked_static': static facts whose evidence is
          NOT yet visible in history.
        - 'available_not_unlocked_dynamic': dynamic facts whose ancestors
          are all used but the inline code hasn't been run.
        - 'artifacts': available artifact nodes.
        """
        available = self.get_available_nodes()

        unlocked_not_articulated: list[dict] = []
        available_static: list[dict] = []
        available_dynamic: list[dict] = []
        artifacts: list[dict] = []

        # First pass: among DAG-available fact nodes, check unlocker satisfaction.
        # Only facts whose preconditions are met can appear here — this prevents
        # showing child facts (e.g. f12 depends_on f2) before their parents
        # have been explored.
        unlocked_ids: set[str] = set()
        for node in available:
            nid = node['id']
            if not self._is_fact_node(node):
                continue
            fact_type = node.get('type', '')
            if fact_type == 'dynamic':
                continue
            if self.check_fact_unlocker_satisfied(nid, history_text):
                unlocked_not_articulated.append(node)
                unlocked_ids.add(nid)

        # Update the persistent set so mark_used() can bypass preconditions
        self._unlocker_satisfied = unlocked_ids

        # Second pass: precondition-gated facts NOT in unlocked set
        for node in available:
            nid = node['id']
            ntype = node.get('node_type', '')

            if ntype in _ARTIFACT_TYPES:
                artifacts.append(node)
                continue

            if not self._is_fact_node(node):
                continue

            if nid in unlocked_ids:
                continue

            fact_type = node.get('type', '')
            if fact_type == 'dynamic':
                available_dynamic.append(node)
                continue

            available_static.append(node)

        return {
            'unlocked_not_articulated': unlocked_not_articulated,
            'available_not_unlocked_static': available_static,
            'available_not_unlocked_dynamic': available_dynamic,
            'artifacts': artifacts,
        }

    def get_unused_fact_statements(self) -> list[dict]:
        """Return ID + statement of all unused fact-type nodes."""
        result: list[dict] = []
        for nid, node in self._nodes.items():
            if self._is_fact_node(node) and not self._used[nid]:
                result.append({
                    'id': nid,
                    'statement': node.get('statement', ''),
                })
        return result

    def skip_remaining_facts(self) -> list[str]:
        """Mark all unused fact nodes as used (skipped)."""
        skipped: list[str] = []
        for nid, node in self._nodes.items():
            if self._is_fact_node(node) and not self._used[nid]:
                self._used[nid] = True
                self._used_at_step[nid] = -1
                skipped.append(nid)
        if skipped:
            logger.info(
                f'[FactTrackerV2] Skipped {len(skipped)} remaining '
                f'facts: {skipped}'
            )
        return skipped
