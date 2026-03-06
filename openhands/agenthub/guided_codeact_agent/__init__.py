from openhands.agenthub.guided_codeact_agent.guided_codeact_agent import (
    GuidedCodeActAgent,
)
from openhands.controller.agent import Agent

Agent.register('GuidedCodeActAgent', GuidedCodeActAgent)
