"""Fact tracker for v6 investigation graphs.

Parses ``stage2_facts.json`` files with node types: ``fact``,
``reproduce_script``, ``issue_analysis``, ``fix_plan``, ``code_edit``,
``validation``.  Maintains a boolean usage map and DAG-based availability.
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

# Oracle planner stage constants
STAGE_EXPLORATION = 'exploration'
STAGE_REPRODUCTION = 'reproduction'
STAGE_ANALYSIS_PLANNING = 'analysis_planning'
STAGE_IMPLEMENTATION_VERIFICATION = 'implementation_verification'
STAGE_FINISH = 'finish'


class FactTracker:
    """Manages v6 investigation facts with DAG-gated availability."""

    def __init__(self, data: dict) -> None:
        self._nodes: dict[str, dict] = {}
        self._used: dict[str, bool] = {}
        self._used_at_step: dict[str, int] = {}  # node_id -> step when marked used
        self._children: dict[str, list[str]] = {}  # parent -> children
        self._unlocker_satisfied: set[str] = set()  # fact IDs whose unlocker is met
        self._all_impl_done_at_step: int | None = None  # step when all edits+validations first completed
        self._finish_extension_budget: int = 10  # max extra steps after all impl done

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
            f'[FactTracker] Loaded {len(self._nodes)} nodes: '
            f'{len(self._ps_roots)} PS-roots, {len(self._non_ps_roots)} non-PS-roots'
        )

        # Auto-mark problem_statement roots as used — the issue text is
        # always visible in the prompt so these are satisfied from step 0.
        for nid in self._ps_roots:
            self._used[nid] = True
            self._used_at_step[nid] = 0
        if self._ps_roots:
            logger.info(
                f'[FactTracker] Auto-marked {len(self._ps_roots)} '
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
        """Return True if this node is a fact-level node (not an artifact).

        Handles both standard ``node_type='fact'`` and variant types like
        ``'dynamic'`` that should be treated as facts.  Any node whose
        ``node_type`` is not one of the known artifact types is considered
        a fact.
        """
        return node.get('node_type', '') not in _ARTIFACT_TYPES

    def _depends_on_artifact(self, node_id: str) -> bool:
        """Return True if the node directly or transitively depends on an artifact.

        Some fact graphs have facts that depend on ``repro1`` or other
        artifacts.  These facts cannot be satisfied during exploration
        and must be deferred to later stages.
        """
        visited: set[str] = set()

        def _walk(nid: str) -> bool:
            if nid in visited:
                return False
            visited.add(nid)
            node = self._nodes.get(nid)
            if not node:
                return False
            if node.get('node_type', '') in _ARTIFACT_TYPES:
                return True
            for dep in node.get('depends_on', []):
                if _walk(dep):
                    return True
            return False

        for dep in self._nodes.get(node_id, {}).get('depends_on', []):
            if _walk(dep):
                return True
        return False

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def _preconditions_met(self, node_id: str, relaxed_artifacts: bool = False) -> bool:
        """Check if all depends_on nodes are used.

        If ``relaxed_artifacts`` is True, inter-artifact dependencies
        (edit→edit, edit→validation) are skipped — only non-artifact
        (fact-level) dependencies are enforced.  This allows the oracle
        to apply edits and validations in any natural order during the
        implementation stage.
        """
        deps = self._nodes[node_id].get('depends_on', [])
        for d in deps:
            if self._used.get(d, False):
                continue  # satisfied
            if relaxed_artifacts:
                dep_node = self._nodes.get(d)
                if dep_node and dep_node.get('node_type') in _ARTIFACT_TYPES:
                    continue  # skip inter-artifact dep in relaxed mode
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

    def mark_used(self, node_ids: list[str], step_index: int = -1) -> None:
        current_stage = self.get_current_stage()
        past_exploration = current_stage != STAGE_EXPLORATION

        for nid in node_ids:
            # Resolve: if nid is unknown, try to match it as a node_type
            # (planner LLMs sometimes output "reproduce_script" instead of "repro1")
            if nid not in self._used:
                resolved = self._resolve_node_id(nid)
                if resolved:
                    logger.info(
                        f'[FactTracker] Resolved "{nid}" → "{resolved}" '
                        f'(matched by node_type).'
                    )
                    nid = resolved
                else:
                    logger.warning(f'[FactTracker] Unknown node ID: {nid}')
                    continue
            if self._used[nid]:
                continue  # already used, skip silently

            node = self._nodes.get(nid, {})
            ntype = node.get('node_type', '')

            # Past exploration: ALL nodes (artifacts and facts) are marked
            # unconditionally.  No precondition checking.  Stage
            # transitions are gated only by whether the relevant
            # artifacts are marked as used.
            if past_exploration:
                self._used[nid] = True
                self._used_at_step[nid] = step_index
                logger.info(
                    f'[FactTracker] Marked {nid} as used '
                    f'(step {step_index}, post-exploration bypass).'
                )
                continue

            # Facts whose unlocker is already satisfied (evidence visible
            # in history) can be marked as used even if their DAG
            # preconditions are not met.  The solver has already seen the
            # evidence and articulated the finding — blocking on
            # preconditions would create a deadlock.
            if nid in self._unlocker_satisfied:
                self._used[nid] = True
                self._used_at_step[nid] = step_index
                logger.info(
                    f'[FactTracker] Marked {nid} as used '
                    f'(step {step_index}, unlocker-satisfied bypass).'
                )
                continue

            # During exploration: enforce preconditions on fact nodes
            if not self._preconditions_met(nid):
                unmet = [
                    d for d in self._nodes[nid].get('depends_on', [])
                    if not self._used.get(d, False)
                ]
                logger.warning(
                    f'[FactTracker] Cannot mark {nid} as used: '
                    f'unmet preconditions {unmet}. Skipping.'
                )
                continue
            self._used[nid] = True
            self._used_at_step[nid] = step_index
            logger.info(f'[FactTracker] Marked {nid} as used (step {step_index}).')

    def _resolve_node_id(self, candidate: str) -> str | None:
        """Try to resolve an unknown ID by matching against node_type.

        The planner LLM sometimes outputs ``"reproduce_script"`` when the
        actual node ID is ``"repro1"``, or ``"issue_analysis"`` instead of
        ``"analysis"``.  This fallback handles that gracefully.
        """
        # Exact node_type match → return first matching node ID
        for nid, node in self._nodes.items():
            if node.get('node_type') == candidate:
                return nid
        # Partial match: strip common prefixes/suffixes
        # e.g., "code_edit_exception" → match node_type "code_edit"
        for nid, node in self._nodes.items():
            ntype = node.get('node_type', '')
            if ntype and (candidate.startswith(ntype) or ntype.startswith(candidate)):
                return nid
        return None

    def is_used(self, node_id: str) -> bool:
        return self._used.get(node_id, False)

    # ------------------------------------------------------------------
    # Phase gate checking
    # ------------------------------------------------------------------

    def get_nodes_by_type(self, node_type: str) -> list[dict]:
        """Return all nodes of a given type."""
        return [n for n in self._nodes.values() if n.get('node_type') == node_type]

    def get_unmet_dependencies(self, node_id: str) -> list[str]:
        """Return list of direct dependency IDs that are not yet used."""
        if node_id not in self._nodes:
            return []
        deps = self._nodes[node_id].get('depends_on', [])
        return [d for d in deps if not self._used.get(d, False)]

    def get_blocking_ancestors(self, node_id: str) -> list[str]:
        """Return all ancestor fact IDs (recursively) that are not yet used.

        These are the specific facts that must be unlocked before the given
        node can be marked as used.  Returns only *leaf* blockers — i.e.,
        ancestors whose own dependencies are all satisfied (available to
        unlock right now).
        """
        if node_id not in self._nodes:
            return []

        blockers: list[str] = []
        visited: set[str] = set()

        def _walk(nid: str) -> None:
            if nid in visited:
                return
            visited.add(nid)
            if self._used.get(nid, False):
                return  # already satisfied
            # Check if this node's own deps are all met → it's a leaf blocker
            deps = self._nodes.get(nid, {}).get('depends_on', [])
            unmet_deps = [d for d in deps if not self._used.get(d, False)]
            if not unmet_deps:
                # This node is available but not used — it's a leaf blocker
                if nid != node_id:  # don't include the node itself
                    blockers.append(nid)
            else:
                # Recurse into unmet deps
                for d in unmet_deps:
                    _walk(d)

        for dep in self._nodes[node_id].get('depends_on', []):
            _walk(dep)
        return blockers

    def check_phase_readiness(self) -> dict[str, tuple[bool, list[str]]]:
        """Check which phases can be entered based on artifact usage.

        Post-exploration, only checks whether the relevant artifacts are
        marked as used — no DAG dependency checking.

        Returns dict mapping phase key to ``(ready, blocking_reasons)``.
        Phase keys: TEST_CREATION, FIX_ANALYSIS, FIX_IMPLEMENTATION, VERIFICATION.
        """
        result: dict[str, tuple[bool, list[str]]] = {}

        # --- Phase 4 (TEST CREATION): always ready post-exploration
        result['TEST_CREATION'] = (True, [])

        # --- Phase 5 (FIX ANALYSIS): reproduce_script must be used
        repro_nodes = self.get_nodes_by_type('reproduce_script')
        repro_used = any(self._used.get(rn['id'], False) for rn in repro_nodes) if repro_nodes else True
        if not repro_used:
            result['FIX_ANALYSIS'] = (
                False,
                ['The reproduce_script artifact is not yet completed. '
                 'A reproduction script must be created and run before analysis can begin.'],
            )
        else:
            result['FIX_ANALYSIS'] = (True, [])

        # --- Phase 6 (FIX IMPLEMENTATION): analysis + plan must be used
        analysis_nodes = self.get_nodes_by_type('issue_analysis')
        plan_nodes = self.get_nodes_by_type('fix_plan')
        impl_blocking: list[str] = []
        if analysis_nodes and not any(self._used.get(n['id'], False) for n in analysis_nodes):
            impl_blocking.append(
                'The issue_analysis artifact is not yet completed. '
                'Use the `think` tool to analyze the root cause before implementing.'
            )
        if plan_nodes and not any(self._used.get(n['id'], False) for n in plan_nodes):
            impl_blocking.append(
                'The fix_plan artifact is not yet completed. '
                'Use the `think` tool to plan the fix before implementing.'
            )
        result['FIX_IMPLEMENTATION'] = (len(impl_blocking) == 0, impl_blocking)

        # --- Phase 7 (VERIFICATION): all code_edit must be used
        edit_nodes = self.get_nodes_by_type('code_edit')
        unused_edits = [n['id'] for n in edit_nodes if not self._used.get(n['id'], False)]
        if unused_edits:
            result['VERIFICATION'] = (
                False,
                [f'Code edit artifacts not yet completed: {unused_edits}. '
                 f'All code changes must be applied before running verification tests.'],
            )
        else:
            result['VERIFICATION'] = (True, [])

        return result

    # ------------------------------------------------------------------
    # Phase progress
    # ------------------------------------------------------------------

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
            # For fact nodes, show statement; for artifacts, show type + description
            if self._is_fact_node(node):
                summary = node.get('statement', '')
            else:
                summary = node.get('description', node.get('text', ntype))
            # Truncate long summaries
            if len(summary) > 200:
                summary = summary[:197] + '...'
            lines.append(f'- **[{nid}]** (step {step}, {ntype}): {summary}')

        return '\n'.join(lines)

    def render_available_facts_for_planner(self) -> str:
        """Render available facts grouped by type for the planner prompt."""
        available = self.get_available_nodes()
        if not available:
            return '(No facts currently available — all are either used or blocked by unmet preconditions.)'

        ps_root_facts = []
        non_ps_root_facts = []
        regular_facts = []
        artifacts = []

        for node in available:
            nid = node['id']
            ntype = node.get('node_type', '')
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

    def _render_fact_node(self, node: dict) -> str:
        nid = node['id']
        ntype = node.get('type', 'unknown')  # static / dynamic
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

    # ------------------------------------------------------------------
    # Render used facts summary for planner context
    # ------------------------------------------------------------------

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
    # Stage determination
    # ------------------------------------------------------------------

    def all_facts_used(self) -> bool:
        """Return True if all exploration-phase facts are used.

        Facts that depend (directly or transitively) on artifacts like
        ``repro1`` are excluded — they cannot be satisfied during
        exploration and are deferred to later stages.
        """
        for nid, node in self._nodes.items():
            if self._is_fact_node(node) and not self._used[nid]:
                if self._depends_on_artifact(nid):
                    continue  # deferred to post-exploration
                return False
        return True

    def skip_remaining_facts(self) -> list[str]:
        """Mark all unused fact nodes as used (skipped).

        Called when the critic confirms remaining facts are unrelated
        to the issue and exploration can be exited early.

        Returns the list of skipped fact IDs.
        """
        skipped: list[str] = []
        for nid, node in self._nodes.items():
            if self._is_fact_node(node) and not self._used[nid]:
                if self._depends_on_artifact(nid):
                    continue  # deferred to post-exploration, don't skip
                self._used[nid] = True
                self._used_at_step[nid] = -1  # sentinel for "skipped"
                skipped.append(nid)
        if skipped:
            logger.info(
                f'[FactTracker] Skipped {len(skipped)} remaining '
                f'facts as unrelated: {skipped}'
            )
        return skipped

    def get_unused_fact_statements(self) -> list[dict]:
        """Return ID + statement of unused exploration-phase facts.

        Excludes facts that depend on artifacts (deferred to later stages).
        """
        result: list[dict] = []
        for nid, node in self._nodes.items():
            if self._is_fact_node(node) and not self._used[nid]:
                if self._depends_on_artifact(nid):
                    continue
                result.append({
                    'id': nid,
                    'statement': node.get('statement', ''),
                })
        return result

    def set_finish_extension_budget(self, budget: int) -> None:
        """Set the number of extra steps allowed after all edits+validations done."""
        self._finish_extension_budget = budget

    def get_current_stage(self) -> str:
        """Determine the current oracle planner stage based on fact/artifact usage.

        Stages:
        - EXPLORATION: not all facts are used yet
        - REPRODUCTION: all facts used, reproduce_script not yet done
        - ANALYSIS_PLANNING: repro done, analysis/plan not yet done
        - IMPLEMENTATION_VERIFICATION: analysis+plan done, edits/validation remain
        - FINISH: all edits+validations done AND extension budget exhausted
        """
        progress = self.get_phase_progress()

        if not self.all_facts_used():
            return STAGE_EXPLORATION

        if not progress['repro_done']:
            return STAGE_REPRODUCTION

        if not progress['analysis_done'] or not progress['plan_done']:
            return STAGE_ANALYSIS_PLANNING

        # Check if all implementation artifacts are done
        if progress['all_edits_done'] and progress['all_validations_done']:
            # Track when this first happened
            if self._all_impl_done_at_step is not None:
                return STAGE_FINISH
            # If we just noticed it, the caller will set the step via
            # check_and_set_impl_complete()

        return STAGE_IMPLEMENTATION_VERIFICATION

    def check_and_set_impl_complete(self, current_step: int) -> bool:
        """Check if all edits+validations are done; record the step if so.

        Returns True if a transition to finish countdown was just triggered.
        Should be called after every mark_used() in implementation stage.
        """
        progress = self.get_phase_progress()
        if not (progress['all_edits_done'] and progress['all_validations_done']):
            return False
        if self._all_impl_done_at_step is not None:
            return False  # already set
        self._all_impl_done_at_step = current_step
        logger.info(
            f'[FactTracker] All edits+validations done at step '
            f'{current_step}. Finish extension budget: '
            f'{self._finish_extension_budget} steps.'
        )
        return True

    def should_force_finish(self, current_step: int) -> bool:
        """Return True if the finish extension budget is exhausted."""
        if self._all_impl_done_at_step is None:
            return False
        steps_since = current_step - self._all_impl_done_at_step
        return steps_since >= self._finish_extension_budget

    def get_finish_countdown(self, current_step: int) -> int | None:
        """Return remaining extension steps, or None if not in countdown."""
        if self._all_impl_done_at_step is None:
            return None
        remaining = self._finish_extension_budget - (current_step - self._all_impl_done_at_step)
        return max(0, remaining)

    def render_solving_summary(self) -> str:
        """Render a structured summary of the entire solving trajectory.

        Used in the FINISH stage prompt so the oracle can produce a
        natural conclusion message.
        """
        lines: list[str] = []

        # 1. Issue understanding
        for nid in self._ps_roots:
            node = self._nodes[nid]
            stmt = node.get('statement', '')
            if stmt:
                lines.append(f'**Issue:** {stmt}')

        # 2. Key findings
        lines.append('')
        lines.append('**Key findings during investigation:**')
        for nid, node in self._nodes.items():
            if not self._is_fact_node(node) or not self._used.get(nid):
                continue
            if nid in self._ps_roots:
                continue
            stmt = node.get('statement', '')
            if stmt:
                step = self._used_at_step.get(nid, -1)
                lines.append(f'- [{nid}] (step {step}): {stmt[:200]}')

        # 3. Analysis & Plan
        for ntype_label, ntype in [('Analysis', _ANALYSIS), ('Fix plan', _PLAN)]:
            for nid, node in self._nodes.items():
                if node.get('node_type') == ntype and self._used.get(nid):
                    text = node.get('text', '')
                    if text:
                        lines.append(f'')
                        lines.append(f'**{ntype_label}:** {text[:500]}')

        # 4. Edits applied
        edit_nodes = self.get_nodes_by_type(_EDIT)
        if edit_nodes:
            lines.append('')
            lines.append('**Code changes applied:**')
            for n in edit_nodes:
                if self._used.get(n['id']):
                    f = n.get('file', '?')
                    desc = n.get('description', '')
                    lines.append(f'- [{n["id"]}] {f}: {desc}')

        # 5. Validations
        val_nodes = self.get_nodes_by_type(_VALIDATION)
        if val_nodes:
            lines.append('')
            lines.append('**Validations completed:**')
            for n in val_nodes:
                if self._used.get(n['id']):
                    desc = n.get('description', '')
                    lines.append(f'- [{n["id"]}]: {desc}')

        return '\n'.join(lines)

    def get_stage_visible_artifact_types(self, stage: str) -> set[str]:
        """Return the artifact node types visible in a given stage."""
        if stage == STAGE_EXPLORATION:
            return set()  # no artifacts visible
        elif stage == STAGE_REPRODUCTION:
            return {_REPRO}
        elif stage == STAGE_ANALYSIS_PLANNING:
            return {_REPRO, _ANALYSIS, _PLAN}
        else:  # IMPLEMENTATION_VERIFICATION
            return {_REPRO, _ANALYSIS, _PLAN, _EDIT, _VALIDATION}

    def render_available_facts_for_stage(self, stage: str) -> str:
        """Render available facts filtered by the current stage.

        In each stage, only fact nodes + stage-appropriate artifacts are shown.
        """
        available = self.get_available_nodes()
        if not available:
            return '(No facts currently available — all are either used or blocked by unmet preconditions.)'

        visible_artifact_types = self.get_stage_visible_artifact_types(stage)

        ps_root_facts = []
        non_ps_root_facts = []
        regular_facts = []
        artifacts = []

        for node in available:
            nid = node['id']
            ntype = node.get('node_type', '')
            if ntype == _FACT:
                if nid in self._ps_roots:
                    ps_root_facts.append(node)
                elif nid in self._non_ps_roots:
                    non_ps_root_facts.append(node)
                else:
                    regular_facts.append(node)
            elif ntype in visible_artifact_types:
                artifacts.append(node)
            # else: artifact not visible in this stage — skip

        # If nothing is visible after filtering, report it
        if not ps_root_facts and not non_ps_root_facts and not regular_facts and not artifacts:
            if stage == STAGE_EXPLORATION:
                return '(All fact nodes are either used or blocked. Stage transition imminent.)'
            return '(No nodes currently available for this stage.)'

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

    def get_unexplored_fact_summary(self) -> str:
        """Return a summary of remaining unexplored facts (for stage transition guidance)."""
        unexplored = []
        for nid, node in self._nodes.items():
            if node.get('node_type') == _FACT and not self._used[nid]:
                stmt = node.get('statement', '')
                if len(stmt) > 100:
                    stmt = stmt[:97] + '...'
                unexplored.append(f'  - [{nid}]: {stmt}')
        if not unexplored:
            return ''
        return '\n'.join(unexplored)

    def get_unexplored_fact_breakdown(self) -> dict:
        """Return a breakdown of remaining unexplored facts by status.

        Returns dict with:
        - total: total count of unused facts
        - count_available: facts whose dependencies are met (available to target)
        - count_blocked: facts whose dependencies are not yet met
        """
        total = 0
        available = 0
        blocked = 0
        for nid, node in self._nodes.items():
            if node.get('node_type') == _FACT and not self._used[nid]:
                total += 1
                deps = node.get('depends_on', [])
                if all(self._used.get(d, False) for d in deps):
                    available += 1
                else:
                    blocked += 1
        return {
            'total': total,
            'count_available': available,
            'count_blocked': blocked,
        }

    # ------------------------------------------------------------------
    # Unlocker satisfaction checking
    # ------------------------------------------------------------------

    # Regex for parsing [view] file_path start-end
    _VIEW_PATTERN = re.compile(
        r'\[view\]\s+(\S+?)(?:\s+(\d+)(?:-(\d+))?)?$'
    )

    @classmethod
    def parse_static_unlocker(cls, action: str) -> dict | None:
        """Parse a static unlocker action string.

        Returns dict with keys: type ('view'|'bash'|'problem_statement'),
        and for 'view': file_path, start_line, end_line.
        Returns None if the action cannot be parsed.
        """
        action = action.strip()
        if not action:
            return None

        # Problem statement — always satisfied (the issue text is in the prompt)
        if 'problem_statement' in action.lower():
            return {'type': 'problem_statement'}

        # [view] path start-end
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

        # [bash] ... — not a view unlocker, can't auto-check
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
        """Check whether the interaction history already contains the content
        from the specified file at the specified line range.

        Handles multiple history formats:
        - str_replace_editor view: shows ``cat -n`` output with numbered lines
        - bash cat/head/tail: shows file content (may have line numbers)
        - grep results: shows ``filepath:linenum:content`` matches

        The check is conservative: we require that the file path appears AND
        that at least one line number from the range appears associated with
        file content in the history.
        """
        if not history_text or not file_path:
            return False

        # Normalize — the history may have full workspace paths like
        # /workspace/getmoto__moto__3.0/moto/core/utils.py while the
        # unlocker just says moto/core/utils.py
        # Strategy: check if the file_path suffix appears in the history
        # We escape dots for regex but keep the rest literal
        escaped_path = re.escape(file_path)
        # Match the path at end of a longer absolute path or standalone
        path_pattern = rf'(?:^|[\s/])(?:\S*/)?{escaped_path}'
        if not re.search(path_pattern, history_text, re.MULTILINE):
            return False

        # If no specific lines requested, the file appearing is enough
        if start_line is None:
            return True

        # Check if numbered lines from the range appear in history.
        # The view tool outputs lines like "   415\t..." or "   415  ..."
        # We check if at least one line number from the range appears after
        # the file path mention.
        #
        # Find the position where the file path appears, then check content
        # after that for line numbers from the range.
        path_match = re.search(path_pattern, history_text, re.MULTILINE)
        if not path_match:
            return False

        # Look at content after the file path mention
        remaining = history_text[path_match.start():]

        # Check if any line number from the range appears in a numbered-line
        # context (tab or spaces after the number)
        for line_num in range(start_line, (end_line or start_line) + 1):
            # Match patterns like "  415\t" or "  415  " or ":415:"
            line_pattern = rf'(?:^|\s){line_num}[\t\s:|]'
            if re.search(line_pattern, remaining):
                return True

        return False

    def check_fact_unlocker_satisfied(
        self, node_id: str, history_text: str,
    ) -> bool:
        """Check if a fact's static unlocker is satisfied by the history.

        Returns True if:
        - The fact is a problem_statement root (always satisfied)
        - The fact has a [view] unlocker and the file+lines appear in history
        Returns False if:
        - The fact has a [bash] or unknown unlocker type (can't auto-verify)
        - The view content is not found in history
        """
        node = self._nodes.get(node_id)
        if not node or node.get('node_type') != _FACT:
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

        # bash / other — cannot auto-check
        return False

    def categorize_available_facts(
        self, history_text: str, stage: str,
    ) -> dict[str, list[dict]]:
        """Categorize available fact nodes based on unlocker satisfaction.

        Returns a dict with keys:
        - 'unlocked_not_articulated': facts whose evidence IS visible in
          history but not yet marked as used. The oracle should revise to
          articulate these.  These are included regardless of DAG
          preconditions — if the evidence is already visible, the oracle
          should articulate it even if ancestor facts haven't been marked.
        - 'available_not_unlocked_static': static facts whose evidence is
          NOT yet visible in history. The oracle should target these next.
          Only includes precondition-met facts.
        - 'available_not_unlocked_dynamic': dynamic facts whose ancestors
          are all used but the inline code hasn't been run. The oracle
          should execute these as a complete action.
          Only includes precondition-met facts.
        - 'artifacts': available artifact nodes (filtered by stage).
          Post-exploration, ALL unused stage-appropriate artifacts are
          included regardless of preconditions.
        """
        available = self.get_available_nodes()
        visible_artifact_types = self.get_stage_visible_artifact_types(stage)
        past_exploration = stage != STAGE_EXPLORATION

        unlocked_not_articulated: list[dict] = []
        available_static: list[dict] = []
        available_dynamic: list[dict] = []
        artifacts: list[dict] = []

        # First pass: scan ALL unused fact nodes for unlocker satisfaction.
        # Facts whose evidence is already visible in history should appear
        # in "Unlocker Satisfied" regardless of preconditions.
        unlocked_ids: set[str] = set()
        for nid, node in self._nodes.items():
            if self._used.get(nid, False):
                continue
            if not self._is_fact_node(node):
                continue
            # During exploration, skip facts that depend on artifacts
            # (they are deferred to post-exploration stages)
            if not past_exploration and self._depends_on_artifact(nid):
                continue
            # Only check static facts for unlocker satisfaction
            fact_type = node.get('type', '')
            if fact_type == 'dynamic':
                continue
            if self.check_fact_unlocker_satisfied(nid, history_text):
                unlocked_not_articulated.append(node)
                unlocked_ids.add(nid)

        # Update the persistent set so mark_used() can bypass preconditions
        self._unlocker_satisfied = unlocked_ids

        # Second pass: precondition-gated facts that are NOT already in
        # the unlocked set.
        for node in available:
            nid = node['id']
            ntype = node.get('node_type', '')

            # Skip artifacts here — handled separately below
            if ntype in _ARTIFACT_TYPES:
                if not past_exploration and ntype in visible_artifact_types:
                    artifacts.append(node)
                continue

            if not self._is_fact_node(node):
                continue

            # Skip if already in unlocked set
            if nid in unlocked_ids:
                continue

            # During exploration, skip artifact-dependent facts
            if not past_exploration and self._depends_on_artifact(nid):
                continue

            fact_type = node.get('type', '')
            if fact_type == 'dynamic':
                available_dynamic.append(node)
                continue

            # Static fact with unlocker NOT satisfied
            available_static.append(node)

        # Post-exploration: include ALL unused stage-appropriate artifacts
        # AND all unused artifact-dependent facts regardless of
        # preconditions.  Stage transitions are gated only by whether
        # the relevant artifacts are marked as used.
        if past_exploration:
            for nid, node in self._nodes.items():
                if self._used.get(nid, False):
                    continue
                ntype = node.get('node_type', '')
                if ntype in visible_artifact_types:
                    artifacts.append(node)
                elif self._is_fact_node(node) and nid not in unlocked_ids:
                    # Show artifact-dependent facts as available
                    if self._depends_on_artifact(nid):
                        fact_type = node.get('type', '')
                        if fact_type == 'dynamic':
                            available_dynamic.append(node)
                        else:
                            available_static.append(node)

        return {
            'unlocked_not_articulated': unlocked_not_articulated,
            'available_not_unlocked_static': available_static,
            'available_not_unlocked_dynamic': available_dynamic,
            'artifacts': artifacts,
        }

    def render_implementation_stage_nodes(self) -> str:
        """Render ALL edit and validation nodes for the implementation stage.

        Unlike the standard availability-based rendering, this shows every
        edit and validation node regardless of dependency status. Dependencies
        are presented as recommended sequence / correspondence rather than
        hard gates.
        """
        edit_nodes = self.get_nodes_by_type(_EDIT)
        validation_nodes = self.get_nodes_by_type(_VALIDATION)

        if not edit_nodes and not validation_nodes:
            return '(No edit or validation nodes in the fact graph.)'

        parts: list[str] = []

        # --- Edit nodes ---
        completed_edits = [n for n in edit_nodes if self._used.get(n['id'], False)]
        pending_edits = [n for n in edit_nodes if not self._used.get(n['id'], False)]

        if completed_edits:
            parts.append('### Completed Edits')
            parts.append(
                'These edits have already been applied. Do NOT re-apply them.\n'
            )
            for n in completed_edits:
                step = self._used_at_step.get(n['id'], -1)
                parts.append(f'**[{n["id"]}]** (applied at step {step}) — '
                             f'{n.get("file", "")}')
                desc = n.get('description', '')
                if desc:
                    parts.append(f'  Description: {desc}')
                parts.append('')

        if pending_edits:
            parts.append('### Pending Edits')
            parts.append(
                'These edits need to be applied. The ordering below is the **recommended '
                'sequence** based on logical dependencies, but the oracle may choose a '
                'natural editing order based on the solver\'s current context. Each edit '
                'shows the target file and the exact change to apply.\n'
            )
            for n in pending_edits:
                deps = n.get('depends_on', [])
                # Filter deps to only edit/artifact deps (not facts)
                edit_deps = [d for d in deps if d in self._nodes
                             and self._nodes[d].get('node_type') in _ARTIFACT_TYPES]
                fact_deps = [d for d in deps if d in self._nodes
                             and self._nodes[d].get('node_type') == _FACT]
                parts.append(self._render_artifact_node(n))
                if edit_deps:
                    dep_status = []
                    for d in edit_deps:
                        status = '✓ done' if self._used.get(d, False) else 'pending'
                        parts.append(f'  Recommended after: [{d}] ({status})')
                parts.append('')

        # --- Validation nodes ---
        completed_validations = [n for n in validation_nodes if self._used.get(n['id'], False)]
        pending_validations = [n for n in validation_nodes if not self._used.get(n['id'], False)]

        if completed_validations:
            parts.append('### Completed Validations')
            parts.append(
                'These validation steps have already been performed.\n'
            )
            for n in completed_validations:
                step = self._used_at_step.get(n['id'], -1)
                parts.append(f'**[{n["id"]}]** (completed at step {step}) — '
                             f'{n.get("description", "")}')
                parts.append('')

        if pending_validations:
            parts.append('### Pending Validations')
            parts.append(
                'These validation steps verify the edits. The dependency list shows which '
                'edits each validation corresponds to — run the validation after its '
                'corresponding edits are applied. The oracle should select or produce '
                'natural validation actions (running tests, reproduction scripts, etc.).\n'
            )
            for n in pending_validations:
                parts.append(self._render_artifact_node(n))
                deps = n.get('depends_on', [])
                edit_deps = [d for d in deps if d in self._nodes
                             and self._nodes[d].get('node_type') == _EDIT]
                if edit_deps:
                    dep_status = []
                    for d in edit_deps:
                        status = '✓ done' if self._used.get(d, False) else 'pending'
                        dep_status.append(f'[{d}] ({status})')
                    parts.append(f'  Corresponds to edits: {", ".join(dep_status)}')
                parts.append('')

        return '\n'.join(parts)

    def render_categorized_facts(
        self, history_text: str, stage: str,
    ) -> str:
        """Render available facts with unlocker-aware categorization."""

        # In implementation stage, show all edit/validation nodes upfront
        # plus any remaining fact nodes that are still available
        if stage == STAGE_IMPLEMENTATION_VERIFICATION:
            parts: list[str] = []

            # Still show any remaining unlocked-but-not-articulated facts
            cats = self.categorize_available_facts(history_text, stage)
            unlocked = cats['unlocked_not_articulated']
            if unlocked:
                parts.append('### Unlocker Satisfied — Needs Articulation')
                parts.append(
                    'The unlocker action for these facts has already been performed — the '
                    'relevant code or output is visible in the interaction history. However, '
                    'the solver has not yet written reasoning that interprets the finding. '
                    'You MUST revise the candidate to add reasoning that articulates what '
                    'was discovered, then mark these nodes as used.\n'
                )
                for n in unlocked:
                    parts.append(self._render_fact_node(n))

            # Show ALL edit and validation nodes
            impl_text = self.render_implementation_stage_nodes()
            parts.append(impl_text)
            return '\n'.join(parts)

        cats = self.categorize_available_facts(history_text, stage)

        unlocked = cats['unlocked_not_articulated']
        static_avail = cats['available_not_unlocked_static']
        dynamic_avail = cats['available_not_unlocked_dynamic']
        artifacts = cats['artifacts']

        if not unlocked and not static_avail and not dynamic_avail and not artifacts:
            if stage == STAGE_EXPLORATION:
                return '(All fact nodes are either used or blocked. Stage transition imminent.)'
            return '(No nodes currently available for this stage.)'

        parts: list[str] = []

        # Section 1: Unlocked but not yet articulated
        if unlocked:
            parts.append('### Unlocker Satisfied — Needs Articulation')
            parts.append(
                'The unlocker action for these facts has already been performed — the '
                'relevant code or output is visible in the interaction history. However, '
                'the solver has not yet written reasoning that interprets the finding. '
                'You MUST revise the candidate to add reasoning that articulates what '
                'was discovered, then mark these nodes as used. A single revision can '
                'cover multiple nodes if the reasoning flows naturally.\n'
            )
            for n in unlocked:
                parts.append(self._render_fact_node(n))

        # Section 2: Available static facts (not yet unlocked)
        # Split into non-PS roots vs regular facts for different guidance
        non_ps_static = [n for n in static_avail if n['id'] in self._non_ps_roots]
        regular_static = [n for n in static_avail if n['id'] not in self._non_ps_roots]

        if non_ps_static:
            parts.append('### Non-Problem-Statement Root Facts — Unlocker Not Yet Satisfied')
            parts.append(
                'These facts have no DAG preconditions but their unlockers reference '
                'code/files NOT mentioned in the problem statement. You can target '
                'them only when:\n'
                '  (a) The solver has already discovered the relevant file/code area '
                'in the interaction history, OR\n'
                '  (b) You guide the solver to naturally explore that code area with a '
                'logically connected rationale from the current investigation.\n'
                'Do NOT randomly suggest viewing an unrelated file.\n'
            )
            for n in non_ps_static:
                parts.append(self._render_fact_node(n))

        if regular_static:
            parts.append('### Available — Unlocker Not Yet Satisfied')
            parts.append(
                'These facts have all preconditions met but their unlocker action has '
                'not been performed yet — the relevant code/output is not in the '
                'interaction history. Choose your next target from here and guide the '
                'solver to perform the unlocker action (e.g., view the file, run the '
                'command) that reveals the information.\n'
            )
            for n in regular_static:
                parts.append(self._render_fact_node(n))

        # Section 3: Available dynamic facts
        if dynamic_avail:
            parts.append('### Available Dynamic — Unlocker Requires Execution')
            parts.append(
                'These facts have a [bash] unlocker that requires running inline code. '
                'Their ancestors are all used. When targeting these, the response MUST '
                'include both reasoning text AND the bash execution in a single '
                'response. Use a natural transition like "Let me verify this by running '
                'a quick '
                'test..." before the tool call.\n'
            )
            for n in dynamic_avail:
                parts.append(self._render_fact_node(n))

        # Section 4: Artifacts
        if artifacts:
            parts.append('### Available Artifact Nodes')
            parts.append('These artifacts have all preconditions met and are ready to be produced.\n')
            for n in artifacts:
                parts.append(self._render_artifact_node(n))

        return '\n'.join(parts)
