from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from jinja2 import Environment, FileSystemLoader

from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM
from openhands.llm.metrics import Metrics


@dataclass
class PlannerDecision:
    step_index: int
    decision: str  # candidate | proposal
    best_candidate_index: int
    chosen_candidate_index: int | None
    reason: str
    proposal_response_text: str
    raw_planner_response: str
    referenced_fact_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'step_index': self.step_index,
            'decision': self.decision,
            'best_candidate_index': self.best_candidate_index,
            'chosen_candidate_index': self.chosen_candidate_index,
            'reason': self.reason,
            'proposal_response_text': self.proposal_response_text,
            'raw_planner_response': self.raw_planner_response,
            'referenced_fact_ids': self.referenced_fact_ids,
        }


class ReactFactTracker:
    """Tracks structured facts and their usage state across planner steps.

    Supports three data formats:
    1. **Bridged graph** (swegym_v5 ``stage3_bridged.json``): a DAG of nodes with
       categories (fact, bridge_fact, organizational_fact, plan_fact,
       edit_step, validation_step), each with a **single** evidence dict
       ``{action, observation}`` and a ``motivation`` field.
    2. **Stage-2 graph** (swegym_v3 ``stage2_facts.json``): a DAG of nodes with
       categories (trigger, base_fact, organizational_fact, plan_fact,
       edit_step, validation_step), each with an **array** of evidence items.
    3. **Legacy stage-based** (``_react_facts.json``): flat stages with facts.

    Usage tracking (graph mode):
    - A node is ``not_used`` until it has been referenced by the planner.
    - A node is ``fully_used`` once referenced (since v5 evidence is a single item).
    - Fully used nodes are **omitted** from the planner input.
    - A node is only **available** (shown to the planner) when:
      (a) it has no preconditions, OR
      (b) ALL of its precondition nodes are ``fully_used``.
    - This enforces DAG ordering: downstream nodes only become visible
      once their upstream dependencies have been fully consumed.

    Planner references facts using node IDs (e.g. ``"f1"``, ``"b2"``, ``"e3"``).
    """

    # Categories ordered by investigation workflow
    _CATEGORY_ORDER = [
        'fact',
        'bridge_fact',
        'organizational_fact',
        'plan_fact',
        'edit_step',
        'validation_step',
        # Legacy v3 categories (kept for backward compat)
        'trigger',
        'base_fact',
    ]

    _CATEGORY_LABELS = {
        'fact': 'Facts',
        'bridge_fact': 'Bridge Facts (discovered during investigation)',
        'organizational_fact': 'Organizational Facts (synthesis)',
        'plan_fact': 'Plan',
        'edit_step': 'Edit Steps',
        'validation_step': 'Validation Steps',
        # Legacy v3 labels
        'trigger': 'Triggers (initial observations)',
        'base_fact': 'Base Facts (established knowledge)',
    }

    # Categories that represent investigation (must be consumed before implementation)
    _INVESTIGATION_CATEGORIES = {'fact', 'bridge_fact', 'trigger', 'base_fact'}
    # Categories that represent implementation (unlocked after investigation is done)
    _IMPLEMENTATION_CATEGORIES = {'organizational_fact', 'plan_fact', 'edit_step', 'validation_step'}

    def __init__(self, react_facts_data: dict | None) -> None:
        self._nodes: dict[str, dict] = {}
        # Simple boolean usage: True = used, False = not used
        self._used: dict[str, bool] = {}
        self._is_graph_mode = False

        if react_facts_data:
            if 'graph' in react_facts_data:
                self._load_graph(react_facts_data['graph'])
                self._is_graph_mode = True
            elif 'stages' in react_facts_data:
                self._load_legacy(react_facts_data['stages'])

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_graph(self, graph: list[dict]) -> None:
        """Load nodes from graph-based format (stage2 or stage3).

        Handles two evidence shapes:
        - **v5** (stage3_bridged.json): ``evidence`` is a **single dict**
          ``{action, observation}`` — normalised into a 1-element list.
        - **v3** (stage2_facts.json): ``evidence`` is an **array** of
          ``{reasoning, action, observation}`` dicts.
        """
        for node in graph:
            node_id = node.get('id', '')
            if not node_id:
                continue

            raw_evidence = node.get('evidence', [])
            # Normalise evidence to a list for uniform handling
            if isinstance(raw_evidence, dict):
                evidence = [raw_evidence]
            elif isinstance(raw_evidence, list):
                evidence = raw_evidence
            else:
                evidence = []

            self._nodes[node_id] = {
                'id': node_id,
                'category': node.get('category', 'unknown'),
                'statement': node.get('statement', ''),
                'preconditions': node.get('preconditions', []),
                'evidence': evidence,
                # Fields present across versions
                'kind': node.get('kind', ''),
                'title': node.get('title', ''),
                'intention_group': node.get('intention_group', ''),
                'file': node.get('file', ''),
                # v5 fields
                'is_root': node.get('is_root', False),
                'grounding': node.get('grounding', ''),
                'motivation': node.get('motivation', ''),
                'discovery_type': node.get('discovery_type', ''),
            }
            self._used[node_id] = False

        logger.info(
            f'[ReactFactTracker] Loaded graph with {len(self._nodes)} nodes.'
        )

    def _load_legacy(self, stages: list[dict]) -> None:
        """Load from the legacy stages-based format, internally converting
        each fact into a graph-compatible node with a single evidence item."""
        for stage_data in stages:
            stage = stage_data.get('stage', 'unknown')
            goal = stage_data.get('goal', '')
            facts = stage_data.get('facts', [])
            for idx, fact_data in enumerate(facts):
                fact_id = f'{stage}_{idx}'
                rao = fact_data.get('reasoning_action_observation', {})
                evidence = []
                if rao:
                    evidence = [{
                        'reasoning': rao.get('reasoning', ''),
                        'action': rao.get('action', ''),
                        'observation': rao.get('observation', ''),
                    }]
                self._nodes[fact_id] = {
                    'id': fact_id,
                    'category': 'base_fact',
                    'statement': fact_data.get('fact', ''),
                    'preconditions': fact_data.get('preconditions', []),
                    'evidence': evidence,
                    'kind': '',
                    'title': '',
                    'intention_group': '',
                    'file': '',
                    'verification': '',
                    # Legacy fields
                    '_legacy_stage': stage,
                    '_legacy_goal': goal,
                }
                self._used[fact_id] = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_facts(self) -> bool:
        return len(self._nodes) > 0

    @property
    def is_graph_mode(self) -> bool:
        return self._is_graph_mode

    # ------------------------------------------------------------------
    # Node usage state
    # ------------------------------------------------------------------

    def _is_used(self, node_id: str) -> bool:
        """Return True if the node has been consumed."""
        return self._used.get(node_id, True)

    def _preconditions_satisfied(self, node_id: str) -> bool:
        """Check if all precondition nodes of *node_id* are fully_used.

        In graph mode, preconditions are node IDs.  A node is available only
        when every precondition node is ``fully_used``.  Nodes with no
        preconditions are always satisfied.  Unknown precondition IDs are
        treated as satisfied (graceful degradation).
        """
        node = self._nodes.get(node_id)
        if not node:
            return False
        preconditions = node.get('preconditions', [])
        if not preconditions:
            return True
        if not self._is_graph_mode:
            # Legacy format: preconditions are free-text strings, not node IDs.
            # We cannot check them programmatically, so treat as satisfied.
            return True
        for pre_id in preconditions:
            if pre_id not in self._nodes:
                # Unknown node — treat as satisfied to avoid blocking
                continue
            if not self._is_used(pre_id):
                return False
        return True

    def _all_investigation_done(self) -> bool:
        """Return True if all investigation-category nodes have been consumed.

        Investigation categories: fact, bridge_fact (and legacy trigger, base_fact).
        Once all of these are consumed, implementation nodes (organizational_fact,
        plan_fact, edit_step, validation_step) are unlocked without precondition checks.
        """
        for nid, node in self._nodes.items():
            if node['category'] in self._INVESTIGATION_CATEGORIES and not self._is_used(nid):
                return False
        return True

    def get_available_nodes(self) -> list[dict]:
        """Return nodes that are ready for the planner.

        A node is available when:
        1. It is NOT used, AND
        2. Either:
           a. All of its precondition nodes are used (normal DAG gating), OR
           b. All investigation facts are done — in which case ALL remaining
              implementation nodes become available without precondition checks.
        """
        investigation_done = self._all_investigation_done()
        return [
            n for nid, n in self._nodes.items()
            if not self._is_used(nid)
            and (
                investigation_done  # bypass preconditions once investigation is complete
                or self._preconditions_satisfied(nid)
            )
        ]

    def get_all_fact_ids(self) -> list[str]:
        return list(self._nodes.keys())

    # ------------------------------------------------------------------
    # Usage marking
    # ------------------------------------------------------------------

    def mark_facts_used(self, fact_ids: list[str], step_index: int | None = None) -> None:
        """Mark facts as used.  Accepts plain node IDs (e.g. ``"f1"``).

        Already-used nodes are silently skipped to avoid duplicate log noise.
        """
        step_label = f' (step {step_index})' if step_index is not None else ''
        for ref in fact_ids:
            node_id, _ = self._parse_fact_ref(ref)
            if node_id not in self._nodes:
                logger.warning(f'[ReactFactTracker] Unknown node id: {node_id} (from ref "{ref}")')
                continue
            if self._used[node_id]:
                # Already used — skip silently (fixes duplicate-marking bug)
                continue
            self._used[node_id] = True
            logger.info(f'[ReactFactTracker] Marked {node_id} as used{step_label}.')

    @staticmethod
    def _parse_fact_ref(ref: str) -> tuple[str, int | None]:
        """Parse a fact reference like ``"t1:0"`` into ``("t1", 0)``
        or ``"t1"`` into ``("t1", None)``."""
        if ':' in ref:
            parts = ref.rsplit(':', 1)
            try:
                return parts[0], int(parts[1])
            except (ValueError, IndexError):
                return ref, None
        return ref, None

    # ------------------------------------------------------------------
    # Preconditions retrieval
    # ------------------------------------------------------------------

    def get_preconditions_for_facts(self, fact_ids: list[str]) -> list[dict]:
        """Return precondition info for the referenced facts.

        For graph mode, preconditions are node IDs — we resolve them to
        include the statement text for richer context in the critic prompt.
        """
        seen_nodes: set[str] = set()
        result: list[dict] = []

        for ref in fact_ids:
            node_id, _ = self._parse_fact_ref(ref)
            if node_id in seen_nodes or node_id not in self._nodes:
                continue
            seen_nodes.add(node_id)

            node = self._nodes[node_id]
            precondition_ids = node['preconditions']
            statement = node['statement']

            if self._is_graph_mode:
                # Resolve precondition node IDs to their statements
                resolved_preconditions = []
                for pre_id in precondition_ids:
                    pre_node = self._nodes.get(pre_id)
                    if pre_node:
                        resolved_preconditions.append(
                            f'[{pre_id}] ({pre_node["category"]}): '
                            f'{pre_node["statement"][:200]}'
                        )
                    else:
                        resolved_preconditions.append(f'[{pre_id}] (unknown node)')

                result.append({
                    'fact_id': node_id,
                    'category': node['category'],
                    'fact_summary': (
                        statement[:200] + '...'
                        if len(statement) > 200
                        else statement
                    ),
                    'preconditions': resolved_preconditions,
                    'precondition_ids': precondition_ids,
                })
            else:
                # Legacy format: preconditions are already string descriptions
                result.append({
                    'fact_id': node_id,
                    'category': node['category'],
                    'fact_summary': (
                        statement[:200] + '...'
                        if len(statement) > 200
                        else statement
                    ),
                    'preconditions': precondition_ids,
                })

        return result

    # ------------------------------------------------------------------
    # Rendering for planner prompt
    # ------------------------------------------------------------------

    @property
    def all_facts_consumed(self) -> bool:
        """Return True if every node in the graph has been consumed."""
        return all(self._used.values())

    def get_phase_stats(self) -> dict[str, Any]:
        """Return per-phase consumption statistics for graduated guidance.

        Returns a dict with:
        - investigation: {total, used, remaining, done}
        - implementation: {total, used, remaining, done}
        -   edit: {total, used, remaining, done}
        -   validation: {total, used, remaining, done}
        - overall_pct: int (0-100, percentage of all nodes consumed)
        """
        inv_total = inv_used = 0
        impl_total = impl_used = 0
        edit_total = edit_used = 0
        val_total = val_used = 0

        for nid, node in self._nodes.items():
            cat = node['category']
            used = self._is_used(nid)
            if cat in self._INVESTIGATION_CATEGORIES:
                inv_total += 1
                if used:
                    inv_used += 1
            elif cat in self._IMPLEMENTATION_CATEGORIES:
                impl_total += 1
                if used:
                    impl_used += 1
                if cat == 'edit_step':
                    edit_total += 1
                    if used:
                        edit_used += 1
                elif cat == 'validation_step':
                    val_total += 1
                    if used:
                        val_used += 1

        total = len(self._nodes)
        total_used = sum(1 for v in self._used.values() if v)
        overall_pct = int(100 * total_used / total) if total > 0 else 0

        return {
            'investigation': {
                'total': inv_total, 'used': inv_used,
                'remaining': inv_total - inv_used,
                'done': inv_total > 0 and inv_used >= inv_total,
            },
            'implementation': {
                'total': impl_total, 'used': impl_used,
                'remaining': impl_total - impl_used,
                'done': impl_total > 0 and impl_used >= impl_total,
            },
            'edit': {
                'total': edit_total, 'used': edit_used,
                'remaining': edit_total - edit_used,
                'done': edit_total == 0 or edit_used >= edit_total,
            },
            'validation': {
                'total': val_total, 'used': val_used,
                'remaining': val_total - val_used,
                'done': val_total == 0 or val_used >= val_total,
            },
            'overall_pct': overall_pct,
            'total': total,
            'total_used': total_used,
        }

    def render_available_facts_text(self) -> str:
        """Render non-fully-used nodes as structured text for the planner prompt."""
        available = self.get_available_nodes()
        if not available:
            if self.all_facts_consumed:
                return '(All facts have been fully consumed in previous steps.)'
            else:
                return '(No facts currently available — some remain blocked by unsatisfied preconditions.)'

        if self._is_graph_mode:
            return self._render_graph_facts(available)
        else:
            return self._render_legacy_facts(available)

    def _render_graph_facts(self, available: list[dict]) -> str:
        """Render graph-based facts grouped by category."""
        # Group by category
        by_category: dict[str, list[dict]] = {}
        for node in available:
            cat = node['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(node)

        lines: list[str] = []
        for cat in self._CATEGORY_ORDER:
            if cat not in by_category:
                continue
            nodes = by_category[cat]
            label = self._CATEGORY_LABELS.get(cat, cat)
            lines.append(f'### {label}')
            lines.append('')

            for node in nodes:
                node_id = node['id']
                # Nodes shown here are always not-yet-used (available = !used + preconditions met)
                state_marker = ''

                # Header line with ID, optional kind/title
                header = f'**[{node_id}]**{state_marker}'
                if node.get('title'):
                    header += f' — {node["title"]}'
                if node.get('kind'):
                    header += f' ({node["kind"]})'
                if node.get('discovery_type'):
                    header += f' [discovered via: {node["discovery_type"]}]'
                lines.append(header)

                # Statement
                lines.append(f'  Statement: {node["statement"]}')

                # Motivation (v5 field — explains why this fact matters)
                if node.get('motivation'):
                    lines.append(f'  Motivation: {node["motivation"]}')

                # Preconditions (as node ID references)
                if node['preconditions']:
                    lines.append('  Preconditions: ' + ', '.join(
                        f'[{pid}]' for pid in node['preconditions']
                    ))

                # File (for edit_steps)
                if node.get('file'):
                    lines.append(f'  File: {node["file"]}')

                # Evidence (single item per node in v5, possibly array in v3)
                evidence = node.get('evidence', [])
                for ev in evidence:
                    if ev.get('reasoning'):
                        lines.append(f'  Reasoning: {ev["reasoning"]}')
                    if ev.get('action'):
                        lines.append(f'  Action: {ev["action"]}')
                    if ev.get('observation'):
                        lines.append(f'  Expected observation: {ev["observation"]}')

                lines.append('')

        return '\n'.join(lines).strip()

    def _render_legacy_facts(self, available: list[dict]) -> str:
        """Render legacy stage-based facts (backward compat)."""
        by_stage: dict[str, list[dict]] = {}
        for f in available:
            stage = f.get('_legacy_stage', f.get('category', 'unknown'))
            if stage not in by_stage:
                by_stage[stage] = []
            by_stage[stage].append(f)

        lines: list[str] = []
        for stage, facts in by_stage.items():
            goal = facts[0].get('_legacy_goal', '') if facts else ''
            lines.append(f'### Stage: {stage}')
            if goal:
                lines.append(f'Goal: {goal}')
            lines.append('')
            for f in facts:
                fid = f['id']
                lines.append(f'**[{fid}]** {f["statement"]}')
                if f['preconditions']:
                    lines.append('  Preconditions:')
                    for pc in f['preconditions']:
                        lines.append(f'    - {pc}')
                evidence = f.get('evidence', [])
                for eidx, ev in enumerate(evidence):
                    if ev.get('reasoning'):
                        lines.append(f'  Recommended reasoning: {ev["reasoning"]}')
                    if ev.get('action'):
                        lines.append(f'  Recommended action: {ev["action"]}')
                lines.append('')
        return '\n'.join(lines).strip()

    # ------------------------------------------------------------------
    # Usage summary
    # ------------------------------------------------------------------

    def get_usage_summary(self) -> dict:
        total = len(self._nodes)
        used = sum(1 for v in self._used.values() if v)
        not_used = total - used
        investigation_done = self._all_investigation_done()

        available = len(self.get_available_nodes())
        blocked = not_used - available

        return {
            'total_nodes': total,
            'used_nodes': used,
            'not_used_nodes': not_used,
            'available_nodes': available,
            'blocked_nodes': blocked,
            'investigation_done': investigation_done,
            'used_ids': [nid for nid, v in self._used.items() if v],
            # Back-compat fields
            'total_facts': total,
            'used_facts': used,
            'remaining_facts': not_used,
            'used_fact_ids': [nid for nid, v in self._used.items() if v],
        }


class OraclePlanner:
    """Oracle-aware planner that selects a debugger candidate or proposes guidance.

    The planner has access to the golden patch and golden test through
    ``oracle_context``. It must still produce leak-free guidance grounded in the
    current observed history.
    """

    PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'prompts')

    def __init__(
        self,
        llm: LLM,
        issue_text: str,
        oracle_context: str,
        tool_descriptions: str = '',
        react_fact_tracker: ReactFactTracker | None = None,
        prompt_config: Any | None = None,
    ) -> None:
        self.llm = llm
        self.issue_text = issue_text
        self.oracle_context = oracle_context
        self.tool_descriptions = tool_descriptions
        self.react_fact_tracker = react_fact_tracker
        self.prompt_config = prompt_config
        self.max_json_parse_retries = max(
            int(os.environ.get('ORACLE_PLANNER_JSON_PARSE_MAX_RETRIES', '3')),
            0,
        )
        self._jinja_env = Environment(
            loader=FileSystemLoader(self.PROMPTS_DIR),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        # Track accepted decisions for continuity across steps
        self._decision_history: list[dict] = []

    def record_accepted_decision(self, decision: 'PlannerDecision') -> None:
        """Record an accepted planner decision for continuity in future prompts."""
        entry: dict[str, Any] = {
            'step': decision.step_index,
            'decision': decision.decision,
            'reason': decision.reason,
            'consumed_facts': decision.referenced_fact_ids,
        }
        if decision.decision == 'proposal':
            # Extract a meaningful summary from the proposal text
            text = decision.proposal_response_text.strip()
            # Use the first sentence or first 300 chars, whichever is shorter
            dot_pos = text.find('. ')
            if 0 < dot_pos < 300:
                entry['proposal_summary'] = text[:dot_pos + 1]
            else:
                entry['proposal_summary'] = text[:300].rstrip() + ('...' if len(text) > 300 else '')
        self._decision_history.append(entry)

    def _render_decision_history(self, max_entries: int = 8) -> str:
        """Render recent accepted decisions as text for the prompt."""
        if not self._decision_history:
            return ''
        recent = self._decision_history[-max_entries:]
        lines: list[str] = []
        for d in recent:
            consumed = ', '.join(d['consumed_facts']) if d['consumed_facts'] else 'none'
            reason = d['reason']
            if d['decision'] == 'candidate':
                lines.append(f"- **Step {d['step']}** selected candidate (consumed: {consumed})\n  Reason: {reason}")
            else:
                summary = d.get('proposal_summary', '(proposal)')
                lines.append(f"- **Step {d['step']}** proposed (consumed: {consumed})\n  Action: {summary}\n  Reason: {reason}")
        return '\n'.join(lines)

    def plan(
        self,
        step_index: int,
        history_text: str,
        candidates: list[str],
        planner_feedback: str = '',
        attempt: int = 0,
    ) -> PlannerDecision:
        prompt = self._render_prompt(
            step_index=step_index,
            history_text=history_text,
            candidates=candidates,
            planner_feedback=planner_feedback,
        )

        # Log prompt composition breakdown (approx tokens = chars / 4)
        est_tokens = len(prompt) // 4
        section_chars = {
            'issue': len(self.issue_text),
            'oracle_context': len(self.oracle_context),
            'history': len(history_text),
            'candidates': sum(len(c) for c in candidates),
            'facts': len(self.react_fact_tracker.render_available_facts_text()) if self.react_fact_tracker and self.react_fact_tracker.has_facts else 0,
            'tools': len(self.tool_descriptions),
            'feedback': len(planner_feedback),
        }
        section_tokens = {k: v // 4 for k, v in section_chars.items()}
        template_overhead = est_tokens - sum(section_tokens.values())
        section_tokens['template_rules'] = max(template_overhead, 0)

        logger.info(
            f'[OraclePlanner] Step {step_index} prompt: ~{est_tokens} tokens | '
            f'issue={section_tokens["issue"]}, '
            f'oracle={section_tokens["oracle_context"]}, '
            f'history={section_tokens["history"]}, '
            f'candidates={section_tokens["candidates"]}, '
            f'facts={section_tokens["facts"]}, '
            f'tools={section_tokens["tools"]}, '
            f'feedback={section_tokens["feedback"]}, '
            f'template={section_tokens["template_rules"]}'
        )

        for parse_retry in range(self.max_json_parse_retries + 1):
            try:
                response = self.llm.completion(messages=[{'role': 'user', 'content': prompt}])
                raw_text = response.choices[0].message.content or ''
            except Exception as exc:
                logger.warning(
                    f'[OraclePlanner] LLM call failed: {exc}. Falling back to best candidate 0.'
                )
                self._maybe_save_prompt(
                    step_index,
                    attempt,
                    parse_retry,
                    prompt,
                    f'[LLM CALL FAILED]: {exc}',
                )
                return PlannerDecision(
                    step_index=step_index,
                    decision='candidate',
                    best_candidate_index=0,
                    chosen_candidate_index=0,
                    reason=f'Planner LLM call failed: {exc}',
                    proposal_response_text='',
                    raw_planner_response='',
                )

            self._maybe_save_prompt(step_index, attempt, parse_retry, prompt, raw_text)
            parsed = self._parse_response_or_none(step_index, raw_text, len(candidates))
            if parsed is not None:
                return parsed

            if parse_retry < self.max_json_parse_retries:
                logger.warning(
                    '[OraclePlanner] JSON parse failed; retrying planner completion '
                    f'({parse_retry + 1}/{self.max_json_parse_retries}).'
                )

        logger.warning(
            '[OraclePlanner] JSON parse retries exhausted. Falling back to candidate 0.'
        )
        return PlannerDecision(
            step_index=step_index,
            decision='candidate',
            best_candidate_index=0,
            chosen_candidate_index=0,
            reason='Planner response was not parseable JSON after retries.',
            proposal_response_text='',
            raw_planner_response='',
        )

    def _render_prompt(
        self,
        step_index: int,
        history_text: str,
        candidates: list[str],
        planner_feedback: str,
    ) -> str:
        available_facts_text = ''
        has_react_facts = False
        all_facts_consumed = False
        phase_stats = None
        if self.react_fact_tracker and self.react_fact_tracker.has_facts:
            available_facts_text = self.react_fact_tracker.render_available_facts_text()
            has_react_facts = True
            all_facts_consumed = self.react_fact_tracker.all_facts_consumed
            phase_stats = self.react_fact_tracker.get_phase_stats()

        # Resolve prompt section flags from config (with defaults)
        pc = self.prompt_config
        show_tool_descriptions = getattr(pc, 'include_tool_descriptions', True) if pc else True
        show_fact_usage_rules = getattr(pc, 'include_fact_usage_rules', True) if pc else True
        show_finalize_guidance = getattr(pc, 'include_finalize_guidance', True) if pc else True
        show_proposal_format = getattr(pc, 'include_proposal_format', True) if pc else True
        show_workflow_guidelines = getattr(pc, 'include_workflow_guidelines', True) if pc else True

        template = self._jinja_env.get_template('planner_select_or_propose.j2')
        return template.render(
            issue_text=self.issue_text,
            oracle_context=self.oracle_context,
            step_index=step_index,
            history_text=history_text,
            candidates=candidates,
            planner_feedback=planner_feedback,
            tool_descriptions=self.tool_descriptions if show_tool_descriptions else '',
            available_facts_text=available_facts_text,
            has_react_facts=has_react_facts,
            all_facts_consumed=all_facts_consumed,
            show_fact_usage_rules=show_fact_usage_rules,
            show_finalize_guidance=show_finalize_guidance,
            show_proposal_format=show_proposal_format,
            show_workflow_guidelines=show_workflow_guidelines,
            decision_history_text=self._render_decision_history(),
            phase_stats=phase_stats,
        )

    @staticmethod
    def _extract_json(text: str) -> str | None:
        fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if fence:
            return fence.group(1)
        bare = re.search(r'\{.*\}', text, re.DOTALL)
        if bare:
            return bare.group(0)
        return None

    @classmethod
    def _parse_response_or_none(
        cls,
        step_index: int,
        raw_text: str,
        num_candidates: int,
    ) -> PlannerDecision | None:
        json_str = cls._extract_json(raw_text)
        if json_str is None:
            logger.warning('[OraclePlanner] Non-JSON response.')
            return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning(f'[OraclePlanner] Malformed JSON: {exc}.')
            return None

        best_idx = cls._safe_index(data.get('best_candidate_index'), num_candidates)
        decision = str(data.get('decision', 'candidate')).strip().lower()
        chosen_idx_raw = data.get('chosen_candidate_index')
        chosen_idx = cls._safe_index(chosen_idx_raw, num_candidates)
        reason = str(data.get('reason', ''))
        proposal_text = str(data.get('proposal_response', '') or '')
        referenced_fact_ids = [str(x) for x in data.get('referenced_fact_ids', []) if x]

        if decision not in {'candidate', 'proposal'}:
            decision = 'candidate'

        if decision == 'candidate':
            if chosen_idx is None:
                chosen_idx = best_idx
            return PlannerDecision(
                step_index=step_index,
                decision='candidate',
                best_candidate_index=best_idx,
                chosen_candidate_index=chosen_idx,
                reason=reason or 'Planner selected a candidate.',
                proposal_response_text='',
                raw_planner_response=raw_text,
                referenced_fact_ids=referenced_fact_ids,
            )

        if not proposal_text.strip():
            logger.warning(
                '[OraclePlanner] Proposal decision without proposal text. Falling back to best candidate.'
            )
            return PlannerDecision(
                step_index=step_index,
                decision='candidate',
                best_candidate_index=best_idx,
                chosen_candidate_index=best_idx,
                reason=reason or 'Planner proposal empty; fallback to best candidate.',
                proposal_response_text='',
                raw_planner_response=raw_text,
                referenced_fact_ids=referenced_fact_ids,
            )

        return PlannerDecision(
            step_index=step_index,
            decision='proposal',
            best_candidate_index=best_idx,
            chosen_candidate_index=chosen_idx,
            reason=reason or 'Planner proposed a better next response.',
            proposal_response_text=proposal_text,
            raw_planner_response=raw_text,
            referenced_fact_ids=referenced_fact_ids,
        )

    @classmethod
    def _parse_response(
        cls,
        step_index: int,
        raw_text: str,
        num_candidates: int,
    ) -> PlannerDecision:
        parsed = cls._parse_response_or_none(step_index, raw_text, num_candidates)
        if parsed is not None:
            return parsed

        logger.warning('[OraclePlanner] Non-JSON response. Falling back to candidate 0.')
        return PlannerDecision(
            step_index=step_index,
            decision='candidate',
            best_candidate_index=0,
            chosen_candidate_index=0,
            reason='Planner response was not parseable JSON.',
            proposal_response_text='',
            raw_planner_response=raw_text,
        )

    @staticmethod
    def _safe_index(value: Any, num_candidates: int) -> int | None:
        if num_candidates <= 0:
            return 0
        try:
            index = int(value)
        except (TypeError, ValueError):
            return None
        if 0 <= index < num_candidates:
            return index
        return None

    @staticmethod
    def _maybe_save_prompt(
        step_index: int,
        attempt: int,
        parse_retry: int,
        prompt: str,
        raw_response: str,
    ) -> None:
        save_dir = os.environ.get('ORACLE_PLANNER_SAVE_PROMPTS_DIR', '')
        if not save_dir:
            return
        try:
            os.makedirs(save_dir, exist_ok=True)
            if parse_retry == 0:
                fname = f'step_{step_index:04d}_attempt_{attempt:02d}.txt'
            else:
                fname = (
                    f'step_{step_index:04d}_attempt_{attempt:02d}'
                    f'_jsonretry_{parse_retry:02d}.txt'
                )
            fpath = os.path.join(save_dir, fname)
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write('=== ORACLE PLANNER PROMPT ===\n')
                f.write(prompt)
                f.write('\n\n=== ORACLE PLANNER RAW RESPONSE ===\n')
                f.write(raw_response)
                f.write('\n')
        except Exception as exc:
            logger.warning(f'[OraclePlanner] Failed to save prompt debug file: {exc}')

    @classmethod
    def from_env(
        cls,
        issue_text: str,
        oracle_context: str,
        tool_descriptions: str = '',
        react_fact_tracker: ReactFactTracker | None = None,
        prompt_config: Any | None = None,
    ) -> 'OraclePlanner | None':
        from openhands.core.config.utils import get_llm_config_arg

        config_name = os.environ.get('ORACLE_PLANNER_LLM_CONFIG', 'oracle_planner')
        config_file = os.environ.get('CONFIG_FILE', 'config.toml')
        llm_config = get_llm_config_arg(config_name, config_file)
        if llm_config is None:
            logger.warning(
                f'[OraclePlanner] No LLM config found for "{config_name}" in {config_file}. Planner disabled.'
            )
            return None

        llm_config.log_completions = False
        metrics = Metrics(model_name=llm_config.model)
        try:
            llm = LLM(config=llm_config, service_id='oracle_planner', metrics=metrics)
        except Exception as exc:
            logger.warning(f'[OraclePlanner] Failed to initialize planner LLM: {exc}. Planner disabled.')
            return None

        logger.info(
            f'[OraclePlanner] Initialized with model={llm_config.model} config_name={config_name}'
        )
        if react_fact_tracker and react_fact_tracker.has_facts:
            summary = react_fact_tracker.get_usage_summary()
            logger.info(
                f'[OraclePlanner] React facts loaded: {summary["total_facts"]} facts available.'
            )
        return cls(
            llm=llm,
            issue_text=issue_text,
            oracle_context=oracle_context,
            tool_descriptions=tool_descriptions,
            react_fact_tracker=react_fact_tracker,
            prompt_config=prompt_config,
        )
