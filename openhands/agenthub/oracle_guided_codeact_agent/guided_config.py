"""Oracle Guided agent configuration loader.

Reads a YAML config file to control all aspects of the Oracle Guided agent:
oracle context sections, planner settings, critic settings, and debug flags.
Falls back to defaults when no config file is provided or when individual
keys are missing.

Usage:
    from .guided_config import GuidedConfig
    cfg = GuidedConfig.load()              # reads ORACLE_GUIDED_CONFIG env var
    cfg = GuidedConfig.load("path.yaml")   # explicit path
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any

from openhands.core.logger import openhands_logger as logger


# ---------------------------------------------------------------------------
# Nested sections
# ---------------------------------------------------------------------------

@dataclass
class OracleContextConfig:
    """Controls which oracle context components are assembled for the planner."""
    include_golden_patch: bool = True
    include_golden_test_patch: bool = True
    include_issue_understanding: bool = True
    include_react_facts: bool = True


@dataclass
class PlannerConfig:
    """Controls planner behaviour."""
    history_near_window: int = 5  # near interactions: action + observation; older ones: action only
    include_system_instruction: bool = True  # include solver system prompt in planner history
    llm_config: str = 'oracle_planner'
    json_parse_max_retries: int = 3


@dataclass
class CriticConfig:
    """Controls critic behaviour."""
    llm_config: str = 'blinded_critic'
    json_parse_max_retries: int = 3
    enable_symbolic_checks: bool = True  # when False, skip regex extraction + recheck LLM call


@dataclass
class AgentConfig:
    """Controls agent-level behaviour."""
    num_candidates: int = 1
    planner_max_retries: int = 2   # max critic (leakage) retries per step
    gate_max_retries: int = 2      # max stage-gate retries per step
    oracle_start_step: int = 0     # oracle activates at this step (0 = always on)
    oracle_auto_activate: bool = False  # auto-detect Phase 3 header to activate oracle
    oracle_auto_activate_fallback_step: int = 5  # activate oracle after this step if Phase 3 not detected
    finish_extension_steps: int = 10  # extra steps allowed after all edits+validations done
    transient_retries: int = 5     # max retries for transient LLM errors (502, timeout, etc.)
    retry_base_wait: int = 10      # base wait seconds; doubles each retry (10, 20, 40, 80, 160)


@dataclass
class DebugConfig:
    """Controls debug / prompt-saving behaviour."""
    save_planner_prompts: bool = True
    save_critic_prompts: bool = True


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class GuidedConfig:
    """Top-level Oracle Guided configuration."""
    oracle_context: OracleContextConfig = field(default_factory=OracleContextConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    # ---- serialisation helpers ------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if hasattr(value, '__dataclass_fields__'):
                result[f.name] = {
                    sf.name: getattr(value, sf.name) for sf in fields(value)
                }
            else:
                result[f.name] = value
        return result

    # ---- loading --------------------------------------------------------

    @classmethod
    def load(cls, path: str | None = None) -> 'GuidedConfig':
        """Load config from YAML file.

        Resolution order for *path*:
        1. Explicit ``path`` argument.
        2. ``ORACLE_GUIDED_CONFIG`` environment variable.
        3. Return defaults (no file needed).
        """
        if path is None:
            path = os.environ.get('ORACLE_GUIDED_CONFIG', '').strip() or None

        if path is None:
            return cls()

        if not os.path.isfile(path):
            logger.warning(
                f'[GuidedConfig] Config file not found: {path}. Using defaults.'
            )
            return cls()

        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                raw = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning(
                f'[GuidedConfig] Failed to load config from {path}: {exc}. '
                'Using defaults.'
            )
            return cls()

        logger.info(f'[GuidedConfig] Loaded config from {path}')
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> 'GuidedConfig':
        return cls(
            oracle_context=_populate(OracleContextConfig, raw.get('oracle_context', {})),
            planner=_populate(PlannerConfig, raw.get('planner', {})),
            critic=_populate(CriticConfig, raw.get('critic', {})),
            agent=_populate(AgentConfig, raw.get('agent', {})),
            debug=_populate(DebugConfig, raw.get('debug', {})),
        )

    # ---- env-var export -------------------------------------------------

    def export_to_env(self) -> None:
        """Set environment variables from config values.

        Env vars already set take precedence — this only fills in missing ones.
        """
        _setdefault('GUIDED_NUM_CANDIDATES', str(self.agent.num_candidates))
        _setdefault('GUIDED_PLANNER_MAX_RETRIES', str(self.agent.planner_max_retries))
        _setdefault('GUIDED_GATE_MAX_RETRIES', str(self.agent.gate_max_retries))
        _setdefault('GUIDED_ORACLE_START_STEP', str(self.agent.oracle_start_step))
        _setdefault('GUIDED_ORACLE_AUTO_ACTIVATE', '1' if self.agent.oracle_auto_activate else '0')
        _setdefault('GUIDED_ORACLE_AUTO_ACTIVATE_FALLBACK_STEP',
                     str(self.agent.oracle_auto_activate_fallback_step))
        _setdefault('GUIDED_PLANNER_HISTORY_NEAR_WINDOW', str(self.planner.history_near_window))
        _setdefault('GUIDED_PLANNER_LLM_CONFIG', self.planner.llm_config)
        _setdefault('GUIDED_CRITIC_LLM_CONFIG', self.critic.llm_config)
        _setdefault('GUIDED_PLANNER_JSON_PARSE_MAX_RETRIES',
                     str(self.planner.json_parse_max_retries))
        _setdefault('GUIDED_CRITIC_JSON_PARSE_MAX_RETRIES',
                     str(self.critic.json_parse_max_retries))
        _setdefault('GUIDED_CRITIC_ENABLE_SYMBOLIC_CHECKS',
                     '1' if self.critic.enable_symbolic_checks else '0')
        _setdefault('GUIDED_SAVE_PLANNER_PROMPTS',
                     '1' if self.debug.save_planner_prompts else '0')
        _setdefault('GUIDED_SAVE_CRITIC_PROMPTS',
                     '1' if self.debug.save_critic_prompts else '0')
        _setdefault('GUIDED_TRANSIENT_RETRIES', str(self.agent.transient_retries))
        _setdefault('GUIDED_RETRY_BASE_WAIT', str(self.agent.retry_base_wait))
        logger.info('[GuidedConfig] Env vars exported (only missing ones filled).')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _populate(cls: type, data: dict | None) -> Any:
    """Populate a dataclass from a dict, ignoring unknown keys."""
    if not data:
        return cls()
    known = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in known}
    return cls(**filtered)


def _setdefault(name: str, value: str) -> None:
    """Set env var only if not already present."""
    if name not in os.environ:
        os.environ[name] = value
