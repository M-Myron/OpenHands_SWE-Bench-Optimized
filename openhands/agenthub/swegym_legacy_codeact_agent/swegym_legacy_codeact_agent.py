"""SWE-Gym Legacy CodeAct Agent.

A variant of CodeActAgent that uses the original SWE-Gym paper's tool definitions
and system prompt format for compatibility with the released SWE-Gym SFT trajectories.

This matches the format from: https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litellm import ChatCompletionToolParam

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.agenthub.codeact_agent.tools.swegym_legacy import LEGACY_TOOLS
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm_registry import LLMRegistry


class SweGymLegacyCodeActAgent(CodeActAgent):
    """CodeActAgent using the original SWE-Gym paper's tool and prompt format.

    Key differences from CodeActAgent:
    - Only 3 tools: execute_bash, finish, str_replace_editor
    - execute_bash has only `command` parameter (no is_input, timeout, security_risk)
    - finish has no parameters
    - str_replace_editor has no security_risk parameter
    - System prompt is simpler (no ROLE, EFFICIENCY, etc. sections)
    """

    VERSION = '1.0-swegym-legacy'

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        super().__init__(config, llm_registry)
        # Override tools with legacy versions
        self.tools = self._get_tools()
        logger.info(
            f'SweGymLegacyCodeActAgent initialized with {len(self.tools)} legacy tools: '
            f'{[t["function"]["name"] for t in self.tools]}'
        )

    def _get_tools(self) -> list['ChatCompletionToolParam']:
        """Return the original SWE-Gym paper's tool definitions (3 simple tools)."""
        return list(LEGACY_TOOLS)
