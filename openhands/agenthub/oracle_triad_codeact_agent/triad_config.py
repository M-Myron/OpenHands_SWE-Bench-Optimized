"""Oracle Triad configuration loader.

Reads a YAML config file to control all aspects of the Oracle Triad agent:
oracle context sections, planner prompt sections, agent behaviour, and
verifier settings.  Falls back to defaults when no config file is provided
or when individual keys are missing.

Usage:
    from .triad_config import TriadConfig
    cfg = TriadConfig.load()          # reads ORACLE_TRIAD_CONFIG env var
    cfg = TriadConfig.load("path.yaml")  # explicit path
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
    include_deep_analysis: bool = True
    include_react_facts: bool = True


@dataclass
class PlannerPromptConfig:
    """Controls which sections appear in the planner prompt template."""
    include_tool_descriptions: bool = True
    include_fact_usage_rules: bool = True
    include_finalize_guidance: bool = True
    include_proposal_format: bool = True
    include_workflow_guidelines: bool = True


@dataclass
class AgentConfig:
    """Controls agent-level behaviour."""
    num_candidates: int = 1
    planner_max_retries: int = 2
    planner_history_window: int = 5
    proposal_validator: str = 'verifier'         # 'verifier' | 'critic' | 'none'
    planner_llm_config: str = 'oracle_planner'
    critic_llm_config: str = 'blinded_critic'
    verifier_llm_config: str = ''                # falls back to critic config
    planner_json_parse_max_retries: int = 3
    critic_json_parse_max_retries: int = 3
    verifier_programmatic_only: bool = False
    verifier_extractor_json_retries: int = 2


@dataclass
class DebugConfig:
    """Controls debug / prompt-saving behaviour."""
    save_planner_prompts: bool = True
    save_critic_prompts: bool = True


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class TriadConfig:
    """Top-level Oracle Triad configuration."""
    oracle_context: OracleContextConfig = field(default_factory=OracleContextConfig)
    planner_prompt: PlannerPromptConfig = field(default_factory=PlannerPromptConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    # ---- serialisation helpers ------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (mirrors YAML structure)."""
        result: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if hasattr(value, '__dataclass_fields__'):
                result[f.name] = {
                    sf.name: getattr(value, sf.name)
                    for sf in fields(value)
                }
            else:
                result[f.name] = value
        return result

    # ---- loading --------------------------------------------------------

    @classmethod
    def load(cls, path: str | None = None) -> 'TriadConfig':
        """Load config from a YAML file.

        Resolution order for *path*:
        1. Explicit ``path`` argument.
        2. ``ORACLE_TRIAD_CONFIG`` environment variable.
        3. Return defaults (no file needed).

        Missing keys in the YAML silently fall back to defaults.
        """
        if path is None:
            path = os.environ.get('ORACLE_TRIAD_CONFIG', '').strip() or None

        if path is None:
            return cls()

        if not os.path.isfile(path):
            logger.warning(
                f'[TriadConfig] Config file not found: {path}. Using defaults.'
            )
            return cls()

        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                raw = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning(
                f'[TriadConfig] Failed to load config from {path}: {exc}. Using defaults.'
            )
            return cls()

        logger.info(f'[TriadConfig] Loaded config from {path}')
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> 'TriadConfig':
        return cls(
            oracle_context=_populate(OracleContextConfig, raw.get('oracle_context', {})),
            planner_prompt=_populate(PlannerPromptConfig, raw.get('planner_prompt', {})),
            agent=_populate(AgentConfig, raw.get('agent', {})),
            debug=_populate(DebugConfig, raw.get('debug', {})),
        )

    # ---- env-var export (for shell-launcher compat) ---------------------

    def export_to_env(self) -> None:
        """Set environment variables from config values.

        This bridges the config file with the existing env-var-based code
        so that both paths (config file vs. env vars) work.  Env vars that
        are already set take precedence — this only fills in missing ones.
        """
        _setdefault('BLINDED_DEBUGGER_NUM_CANDIDATES', str(self.agent.num_candidates))
        _setdefault('ORACLE_PLANNER_MAX_RETRIES', str(self.agent.planner_max_retries))
        _setdefault('ORACLE_PLANNER_HISTORY_WINDOW', str(self.agent.planner_history_window))
        _setdefault('PROPOSAL_VALIDATOR', self.agent.proposal_validator)
        _setdefault('ORACLE_PLANNER_LLM_CONFIG', self.agent.planner_llm_config)
        _setdefault('ORACLE_PROPOSAL_CRITIC_LLM_CONFIG', self.agent.critic_llm_config)
        if self.agent.verifier_llm_config:
            _setdefault('VERIFIER_LLM_CONFIG', self.agent.verifier_llm_config)
        _setdefault('ORACLE_PLANNER_JSON_PARSE_MAX_RETRIES',
                     str(self.agent.planner_json_parse_max_retries))
        _setdefault('ORACLE_PROPOSAL_CRITIC_JSON_PARSE_MAX_RETRIES',
                     str(self.agent.critic_json_parse_max_retries))
        _setdefault('VERIFIER_PROGRAMMATIC_ONLY',
                     '1' if self.agent.verifier_programmatic_only else '0')
        _setdefault('VERIFIER_EXTRACTOR_JSON_RETRIES',
                     str(self.agent.verifier_extractor_json_retries))
        _setdefault('ORACLE_PLANNER_SAVE_PROMPTS',
                     '1' if self.debug.save_planner_prompts else '0')
        _setdefault('ORACLE_PROPOSAL_CRITIC_SAVE_PROMPTS',
                     '1' if self.debug.save_critic_prompts else '0')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _populate(cls: type, raw: dict | None) -> Any:
    """Create a dataclass instance, using *raw* for known fields and defaults
    for anything missing."""
    if not raw:
        return cls()
    known = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in known}
    return cls(**filtered)


def _setdefault(key: str, value: str) -> None:
    """Set env var only if not already set."""
    if key not in os.environ:
        os.environ[key] = value
