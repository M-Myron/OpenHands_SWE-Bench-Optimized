"""Structured interaction history memory for bounded retrieval.

Indexes OpenHands event history into paired action-observation units and provides
keyword / file-path / tag / phase search without external dependencies.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from openhands.events.action import (
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    MessageAction,
)
from openhands.events.action.agent import AgentFinishAction, AgentThinkAction
from openhands.events.action.message import SystemMessageAction
from openhands.events.event import Event
from openhands.events.observation import (
    CmdOutputObservation,
    FileEditObservation,
    FileReadObservation,
    Observation,
)

# ---------------------------------------------------------------------------
# Path / symbol extraction helpers
# ---------------------------------------------------------------------------

_FILE_PATH_RE = re.compile(r'(?:/[\w._-]+){2,}(?:\.\w+)?')
_PYTHON_SYMBOL_RE = re.compile(
    r'\b(?:class|def)\s+(\w+)'                     # class Foo / def bar
    r'|\bisinstance\(\w+,\s*(\w+(?:\.\w+)*)\)'      # G2: isinstance(x, SomeClass)
    r'|(?:^|\s)(\w+(?:\.\w+)*)\s*\('               # G3: Foo.bar(
    r'|\b(\w+Error)\b'                              # G4: ValueError etc.
    r'|:\s*([A-Z]\w*(?:\.\w+)*)'                    # G5: : ClassName or : Module.sub
    r'|=\s*(\w+(?:\.\w+)+)\s*\('                    # G6: = Module.func(
    r'|\b([A-Z]\w*\.\w+(?:\.\w+)*)\b',             # G7: Dotted PascalCase: Query.output_field
)
_TEST_FILE_RE = re.compile(r'(?:^|/)(?:test_\w+|conftest|\w+_test)\.py$', re.IGNORECASE)
# Matches `python <path>/test_*.py` or `python <path>/*_test.py` (running a test script).
_TEST_SCRIPT_CMD_RE = re.compile(
    r'python[23]?\s+\S*(?:test_\w+|\w+_test|test_\w+_\w+)\.py\b', re.IGNORECASE,
)
_REPRO_SCRIPT_RE = re.compile(
    r'(?:^|/)(?:repro|reproduce|reproduction|bug_repro)\w*\.py$', re.IGNORECASE,
)
# Matches Django-style runtests.py invocations (common in SWE-bench Django tasks).
_RUNTESTS_CMD_RE = re.compile(
    r'(?:python[23]?\s+\S*|\./)?\S*runtests\.py\b', re.IGNORECASE,
)
# Matches manage.py test with optional path prefix and python prefix.
_MANAGE_TEST_CMD_RE = re.compile(
    r'(?:python[23]?\s+\S*)?manage\.py\s+test', re.IGNORECASE,
)

# Workflow phase keywords used for heuristic classification.
_PHASE_KEYWORDS: dict[str, list[str]] = {
    'reading': ['reword', 'understand', 'problem statement', 'issue description'],
    'running': ['pytest', 'python -m pytest', 'tox', 'unittest', 'run test', 'test suite', 'runtests.py', 'manage.py test'],
    'exploration': ['grep', 'find', 'rg ', 'ag ', 'ack ', 'cat ', 'head ', 'tail ', 'less '],
    'test_creation': ['reproduce', 'reproduction', 'repro', 'test script'],
    'fix_analysis': [
        'root cause', 'because', 'the bug is', 'the issue is',
        'the problem is', 'should be', 'needs to', 'fix by',
    ],
    'fix_implementation': ['str_replace', 'edit', 'patch'],
    'verification': ['verify', 'verification', 'run test', 'check fix'],
    'final_review': ['diff', 'review', 'compare', 'final'],
}


@dataclass
class HistoryUnit:
    """One retrievable unit from interaction history."""

    unit_id: int
    unit_type: str  # 'base_instruction' | 'initial_user_instruction' | 'action_observation'
    event_indices: list[int] = field(default_factory=list)
    action_type: str | None = None
    action_summary: str = ''
    action_text: str = ''
    observation_text: str = ''
    files_mentioned: list[str] = field(default_factory=list)
    symbols_mentioned: list[str] = field(default_factory=list)
    phase_hint: str | None = None
    tags: set[str] = field(default_factory=set)

    # ---- convenience text for search -----------------------------------------

    @property
    def full_text(self) -> str:
        """Concatenation of action + observation text for keyword search."""
        return f'{self.action_text}\n{self.observation_text}'

    def to_dict(self) -> dict[str, Any]:
        return {
            'unit_id': self.unit_id,
            'unit_type': self.unit_type,
            'event_indices': self.event_indices,
            'action_type': self.action_type,
            'action_summary': self.action_summary,
            'phase_hint': self.phase_hint,
            'tags': sorted(self.tags),
            'files_mentioned': self.files_mentioned,
            'symbols_mentioned': self.symbols_mentioned,
        }


class StructuredHistoryMemory:
    """Indexed interaction history for bounded retrieval.

    Build via ``from_events(events)``, then query through the search methods.
    No external dependencies — keyword scoring uses simple normalised
    term-frequency over the unit's full text.
    """

    def __init__(self, units: list[HistoryUnit]) -> None:
        self.units = units
        self._id_map: dict[int, HistoryUnit] = {u.unit_id: u for u in units}

    # ------------------------------------------------------------------
    # Search / retrieval
    # ------------------------------------------------------------------

    def keyword_search(self, keywords: list[str], top_k: int = 10) -> list[HistoryUnit]:
        """Return units ranked by keyword overlap (case-insensitive)."""
        if not keywords:
            return []
        lower_kws = [kw.lower() for kw in keywords if kw]
        scored: list[tuple[float, HistoryUnit]] = []
        for unit in self.units:
            text = unit.full_text.lower()
            if not text:
                continue
            text_len = len(text) or 1
            score = 0.0
            for kw in lower_kws:
                count = text.count(kw)
                if count > 0:
                    # TF-like score: count normalised by text length, with log damping
                    score += (1 + math.log(count)) / math.log(text_len + 1)
            if score > 0:
                scored.append((score, unit))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [unit for _, unit in scored[:top_k]]

    def file_path_search(self, path_pattern: str) -> list[HistoryUnit]:
        """Return units whose ``files_mentioned`` match *path_pattern* (substring)."""
        pattern_lower = path_pattern.lower()
        results: list[HistoryUnit] = []
        for unit in self.units:
            for fp in unit.files_mentioned:
                if pattern_lower in fp.lower():
                    results.append(unit)
                    break
        return results

    def phase_search(self, phase: str) -> list[HistoryUnit]:
        """Return units whose ``phase_hint`` matches *phase* (case-insensitive)."""
        phase_lower = phase.lower()
        return [u for u in self.units if u.phase_hint and u.phase_hint.lower() == phase_lower]

    def tag_search(self, tags: set[str]) -> list[HistoryUnit]:
        """Return units whose ``tags`` intersect with *tags*."""
        return [u for u in self.units if u.tags & tags]

    def get_unit(self, unit_id: int) -> HistoryUnit | None:
        return self._id_map.get(unit_id)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_all_edited_files(self) -> list[str]:
        """Unique file paths that were edited (in order of first edit)."""
        seen: set[str] = set()
        result: list[str] = []
        for u in self.units:
            if 'edit' in u.tags:
                for fp in u.files_mentioned:
                    if fp not in seen:
                        seen.add(fp)
                        result.append(fp)
        return result

    def get_all_read_files(self) -> list[str]:
        """Unique file paths that were read (in order of first read)."""
        seen: set[str] = set()
        result: list[str] = []
        for u in self.units:
            if 'file_read' in u.tags:
                for fp in u.files_mentioned:
                    if fp not in seen:
                        seen.add(fp)
                        result.append(fp)
        return result

    def get_all_searched_files(self) -> list[str]:
        """File paths appearing in grep / find / search outputs."""
        seen: set[str] = set()
        result: list[str] = []
        for u in self.units:
            if 'grep' in u.tags or 'search' in u.tags:
                for fp in u.files_mentioned:
                    if fp not in seen:
                        seen.add(fp)
                        result.append(fp)
        return result

    def get_all_known_files(self) -> set[str]:
        """Union of edited, read, and searched file paths."""
        files: set[str] = set()
        for u in self.units:
            files.update(u.files_mentioned)
        return files

    def has_think_action(self) -> bool:
        return any('think' in u.tags for u in self.units)

    def has_edit_action(self) -> bool:
        return any('edit' in u.tags for u in self.units)

    def has_test_run_after_edit(self) -> bool:
        """True if there is a test_run unit that comes after any edit unit."""
        last_edit_id = -1
        for u in self.units:
            if 'edit' in u.tags:
                last_edit_id = u.unit_id
        if last_edit_id < 0:
            return False
        return any(
            'test_run' in u.tags and u.unit_id > last_edit_id
            for u in self.units
        )

    def get_latest_phase(self) -> str | None:
        """Return the phase_hint of the last unit that has one."""
        for u in reversed(self.units):
            if u.phase_hint:
                return u.phase_hint
        return None

    def get_completed_phases(self) -> list[str]:
        """Return deduplicated list of phases seen so far, in order."""
        seen: set[str] = set()
        result: list[str] = []
        for u in self.units:
            if u.phase_hint and u.phase_hint not in seen:
                seen.add(u.phase_hint)
                result.append(u.phase_hint)
        return result

    def get_issue_text(self) -> str:
        """Return the initial user instruction text (issue description)."""
        for u in self.units:
            if u.unit_type == 'initial_user_instruction':
                return u.action_text
        return ''

    def summary(self) -> dict[str, Any]:
        """Quick stats about the memory."""
        return {
            'total_units': len(self.units),
            'phases_seen': self.get_completed_phases(),
            'files_read': self.get_all_read_files()[:20],
            'files_edited': self.get_all_edited_files()[:20],
            'has_think': self.has_think_action(),
            'has_edit': self.has_edit_action(),
            'has_test_after_edit': self.has_test_run_after_edit(),
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_events(cls, events: list[Event]) -> 'StructuredHistoryMemory':
        """Build a structured memory from an OpenHands event history list."""
        units: list[HistoryUnit] = []
        unit_id = 0

        # Track which observation indices have been consumed
        # so we can pair action→observation correctly.
        obs_by_cause: dict[int, list[tuple[int, Observation]]] = {}
        for idx, event in enumerate(events):
            if isinstance(event, Observation):
                cause = getattr(event, '_cause', None)
                if cause is None:
                    cause = getattr(event, 'cause', None)
                if cause is not None:
                    obs_by_cause.setdefault(cause, []).append((idx, event))

        consumed_indices: set[int] = set()

        for idx, event in enumerate(events):
            if idx in consumed_indices:
                continue

            # ---- base instruction (SystemMessageAction) ------------------
            if isinstance(event, SystemMessageAction):
                unit = HistoryUnit(
                    unit_id=unit_id,
                    unit_type='base_instruction',
                    event_indices=[idx],
                    action_type='SystemMessageAction',
                    action_summary='SYSTEM: OpenHands system instruction',
                    action_text=event.content or '',
                    tags={'system_instruction'},
                )
                units.append(unit)
                consumed_indices.add(idx)
                unit_id += 1
                continue

            # ---- initial user instruction --------------------------------
            if isinstance(event, MessageAction) and getattr(event, 'source', '') == 'user':
                # Treat the FIRST user message as the initial instruction.
                is_first_user = not any(
                    u.unit_type == 'initial_user_instruction' for u in units
                )
                unit = HistoryUnit(
                    unit_id=unit_id,
                    unit_type='initial_user_instruction' if is_first_user else 'action_observation',
                    event_indices=[idx],
                    action_type='MessageAction',
                    action_summary='USER: issue description' if is_first_user else f'USER: message',
                    action_text=event.content or '',
                    tags={'user_message', 'initial_instruction'} if is_first_user else {'user_message'},
                )
                _enrich_unit(unit, events, idx)
                units.append(unit)
                consumed_indices.add(idx)
                unit_id += 1
                continue

            # ---- agent message (not an action we pair) -------------------
            if isinstance(event, MessageAction) and getattr(event, 'source', '') == 'agent':
                unit = HistoryUnit(
                    unit_id=unit_id,
                    unit_type='action_observation',
                    event_indices=[idx],
                    action_type='MessageAction',
                    action_summary=f'AGENT MSG: {(event.content or "")[:80]}',
                    action_text=event.content or '',
                    tags={'agent_message'},
                )
                _enrich_unit(unit, events, idx)
                units.append(unit)
                consumed_indices.add(idx)
                unit_id += 1
                continue

            # ---- pair-able actions (cmd, file read, file edit, think) -----
            if isinstance(event, (CmdRunAction, FileReadAction, FileEditAction, AgentThinkAction)):
                action_text, summary, tags, action_type_name = _extract_action_info(event)
                obs_text = ''
                paired_indices = [idx]
                consumed_indices.add(idx)

                # Look up paired observation by event.id
                event_id = getattr(event, 'id', None) or getattr(event, '_id', None)
                if event_id is not None and event_id in obs_by_cause:
                    for obs_idx, obs_event in obs_by_cause[event_id]:
                        if obs_idx not in consumed_indices:
                            obs_text_part, obs_tags = _extract_obs_info(obs_event)
                            obs_text += obs_text_part + '\n'
                            tags |= obs_tags
                            paired_indices.append(obs_idx)
                            consumed_indices.add(obs_idx)

                # If no paired observation found by cause, look ahead for the
                # next observation immediately following this action.
                if not obs_text.strip():
                    for look_idx in range(idx + 1, min(idx + 4, len(events))):
                        if look_idx in consumed_indices:
                            continue
                        look_event = events[look_idx]
                        if isinstance(look_event, Observation):
                            obs_text_part, obs_tags = _extract_obs_info(look_event)
                            obs_text += obs_text_part + '\n'
                            tags |= obs_tags
                            paired_indices.append(look_idx)
                            consumed_indices.add(look_idx)
                            break

                unit = HistoryUnit(
                    unit_id=unit_id,
                    unit_type='action_observation',
                    event_indices=sorted(paired_indices),
                    action_type=action_type_name,
                    action_summary=summary,
                    action_text=action_text,
                    observation_text=obs_text.strip(),
                    tags=tags,
                )
                _enrich_unit(unit, events, idx)
                unit.phase_hint = _infer_phase(unit)
                units.append(unit)
                unit_id += 1
                continue

            # ---- agent finish --------------------------------------------
            if isinstance(event, AgentFinishAction):
                unit = HistoryUnit(
                    unit_id=unit_id,
                    unit_type='action_observation',
                    event_indices=[idx],
                    action_type='AgentFinishAction',
                    action_summary='AGENT FINISH',
                    action_text=getattr(event, 'final_thought', '') or getattr(event, 'thought', '') or '',
                    tags={'finish'},
                    phase_hint='final_review',
                )
                units.append(unit)
                consumed_indices.add(idx)
                unit_id += 1
                continue

            # ---- standalone observations (not paired to any action) ------
            if isinstance(event, Observation):
                obs_text_part, obs_tags = _extract_obs_info(event)
                unit = HistoryUnit(
                    unit_id=unit_id,
                    unit_type='action_observation',
                    event_indices=[idx],
                    action_type=None,
                    action_summary=f'OBS: {type(event).__name__}',
                    observation_text=obs_text_part,
                    tags=obs_tags,
                )
                _enrich_unit(unit, events, idx)
                units.append(unit)
                consumed_indices.add(idx)
                unit_id += 1
                continue

            # ---- anything else -------------------------------------------
            consumed_indices.add(idx)
            unit = HistoryUnit(
                unit_id=unit_id,
                unit_type='action_observation',
                event_indices=[idx],
                action_type=type(event).__name__,
                action_summary=f'EVENT: {type(event).__name__}',
                action_text=str(getattr(event, 'content', '') or getattr(event, 'message', '') or ''),
                tags=set(),
            )
            _enrich_unit(unit, events, idx)
            units.append(unit)
            unit_id += 1

        return cls(units)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_action_info(event: Event) -> tuple[str, str, set[str], str]:
    """Return (action_text, summary, tags, action_type_name) for a pair-able action."""
    if isinstance(event, CmdRunAction):
        cmd = event.command or ''
        thought = getattr(event, 'thought', '') or ''
        text = f'{thought}\n{cmd}'.strip() if thought else cmd
        summary = f'RUN: {cmd[:120]}'
        tags: set[str] = {'command'}
        cmd_lower = cmd.lower()
        if any(k in cmd_lower for k in ('grep', 'rg ', 'ag ', 'ack ')):
            tags.add('grep')
            tags.add('search')
        if any(k in cmd_lower for k in ('find ', 'locate ')):
            tags.add('search')
        if any(k in cmd_lower for k in (
            'pytest', 'python -m pytest', 'unittest', 'tox ',
            'runtests.py', 'manage.py test', 'python -m django test',
            'nosetests',
        )):
            tags.add('test_run')
        # Detect running a test script: python <path>/test_*.py or python <path>/*_test.py
        if not tags & {'test_run'} and _TEST_SCRIPT_CMD_RE.search(cmd):
            tags.add('test_run')
        # Detect Django runtests.py and manage.py test with various path prefixes
        if not tags & {'test_run'} and (
            _RUNTESTS_CMD_RE.search(cmd) or _MANAGE_TEST_CMD_RE.search(cmd)
        ):
            tags.add('test_run')
        if any(k in cmd_lower for k in ('cat ', 'head ', 'tail ', 'less ', 'more ')):
            tags.add('file_read')
        if any(k in cmd_lower for k in ('diff ',)):
            tags.add('diff')
        return text, summary, tags, 'CmdRunAction'

    if isinstance(event, FileReadAction):
        path = event.path or ''
        thought = getattr(event, 'thought', '') or ''
        text = f'{thought}\nREAD: {path}'.strip() if thought else f'READ: {path}'
        summary = f'READ: {path}'
        return text, summary, {'file_read'}, 'FileReadAction'

    if isinstance(event, FileEditAction):
        path = getattr(event, 'path', '') or ''
        thought = getattr(event, 'thought', '') or ''
        content = getattr(event, 'content', '') or ''
        old_str = getattr(event, 'old_str', '') or ''
        new_str = getattr(event, 'new_str', '') or ''
        command = getattr(event, 'command', '') or ''
        parts = [thought] if thought else []
        if command:
            parts.append(f'EDIT ({command}): {path}')
        else:
            parts.append(f'EDIT: {path}')
        if old_str:
            parts.append(f'OLD: {old_str}')
        if new_str:
            parts.append(f'NEW: {new_str}')
        if content and not old_str and not new_str:
            parts.append(content)
        text = '\n'.join(parts)
        summary = f'EDIT: {path}'
        tags_set: set[str] = {'edit'}
        if _TEST_FILE_RE.search(path):
            tags_set.add('test_edit')
        return text, summary, tags_set, 'FileEditAction'

    if isinstance(event, AgentThinkAction):
        thought = event.thought or ''
        text = thought
        summary = f'THINK: {thought[:100]}'
        return text, summary, {'think'}, 'AgentThinkAction'

    # Fallback (shouldn't be reached for the types we handle)
    return str(event), f'ACTION: {type(event).__name__}', set(), type(event).__name__


def _extract_obs_info(event: Observation) -> tuple[str, set[str]]:
    """Return (observation_text, tags) for an observation event."""
    tags: set[str] = set()
    content = event.content or ''

    if isinstance(event, CmdOutputObservation):
        exit_code = event.exit_code
        cmd = getattr(event, 'command', '') or ''
        tags.add('cmd_output')
        if exit_code != 0:
            tags.add('error')
        text = f'CMD OUTPUT (exit={exit_code}, cmd={cmd[:80]}):\n{content}'
        return text, tags

    if isinstance(event, FileReadObservation):
        path = getattr(event, 'path', '') or ''
        tags.add('file_content')
        text = f'FILE CONTENT ({path}):\n{content}'
        return text, tags

    if isinstance(event, FileEditObservation):
        path = getattr(event, 'path', '') or ''
        tags.add('edit_result')
        text = f'EDIT RESULT ({path}):\n{content}'
        return text, tags

    # Generic observation
    return content, tags


def _enrich_unit(unit: HistoryUnit, events: list[Event], primary_idx: int) -> None:
    """Extract file paths and symbols from the unit's text."""
    combined = f'{unit.action_text}\n{unit.observation_text}'

    # --- File paths ---
    paths = _FILE_PATH_RE.findall(combined)
    seen: set[str] = set()
    for p in paths:
        if p not in seen:
            seen.add(p)
            unit.files_mentioned.append(p)

    # Also capture explicit path attributes from the event
    event = events[primary_idx] if primary_idx < len(events) else None
    if event is not None:
        for attr in ('path',):
            val = getattr(event, attr, None)
            if val and isinstance(val, str) and val not in seen:
                seen.add(val)
                unit.files_mentioned.append(val)

    # --- Symbols (best effort) ---
    symbols_seen: set[str] = set()
    for m in _PYTHON_SYMBOL_RE.finditer(combined):
        sym = m.group(1) or m.group(2) or m.group(3) or m.group(4) or m.group(5) or m.group(6) or m.group(7)
        if sym and len(sym) > 2 and sym not in symbols_seen:
            # Filter out common noise
            if sym.lower() not in {
                'self', 'cls', 'return', 'print', 'open', 'str', 'int',
                'float', 'bool', 'list', 'dict', 'set', 'tuple', 'len',
                'range', 'type', 'isinstance', 'getattr', 'setattr', 'hasattr',
                'none', 'true', 'false', 'import', 'from', 'not', 'and', 'the',
            }:
                symbols_seen.add(sym)
                unit.symbols_mentioned.append(sym)


def _infer_phase(unit: HistoryUnit) -> str | None:
    """Heuristic: infer the workflow phase of an action-observation unit."""
    tags = unit.tags
    action_type = unit.action_type
    combined_lower = unit.full_text.lower()

    # Think action with substantial analysis → fix_analysis
    if 'think' in tags:
        text = unit.action_text.lower()
        analysis_kws = _PHASE_KEYWORDS['fix_analysis']
        if any(kw in text for kw in analysis_kws) and len(unit.action_text) > 100:
            return 'fix_analysis'
        # Short think → still might be analysis but could be anything
        return 'fix_analysis'

    # File edit on a non-test file → fix_implementation
    if 'edit' in tags and 'test_edit' not in tags:
        return 'fix_implementation'

    # File edit on a test file → test_creation
    if 'test_edit' in tags:
        return 'test_creation'

    # Test run — depends on whether there's been an edit before
    if 'test_run' in tags:
        # We can't check memory context here (no memory ref),
        # so mark as 'running' or 'verification' based on text clues
        verify_kws = _PHASE_KEYWORDS['verification']
        if any(kw in combined_lower for kw in verify_kws):
            return 'verification'
        return 'running'

    # Grep / search → exploration
    if 'grep' in tags or 'search' in tags:
        return 'exploration'

    # File read → exploration
    if 'file_read' in tags:
        # Could be exploration (reading source) or reading test
        return 'exploration'

    # Command with test-creation keywords
    if 'command' in tags:
        if _REPRO_SCRIPT_RE.search(combined_lower):
            return 'test_creation'
        # Check for test-creation patterns: writing a script via cat/echo/tee
        creation_patterns = ['cat >', 'echo ', 'tee ', 'python -c']
        if any(p in combined_lower for p in creation_patterns):
            if any(kw in combined_lower for kw in _PHASE_KEYWORDS['test_creation']):
                return 'test_creation'

    # Diff → final_review
    if 'diff' in tags:
        return 'final_review'

    return None
