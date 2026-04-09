from openhands.agenthub.swegym_legacy_codeact_agent.swegym_legacy_codeact_agent import (
    SweGymLegacyCodeActAgent,
)
from openhands.controller.agent import Agent

Agent.register('SweGymLegacyCodeActAgent', SweGymLegacyCodeActAgent)
