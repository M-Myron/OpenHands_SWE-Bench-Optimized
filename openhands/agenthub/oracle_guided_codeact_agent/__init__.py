from openhands.agenthub.oracle_guided_codeact_agent.oracle_guided_codeact_agent import (
    OracleGuidedCodeActAgent,
)
from openhands.controller.agent import Agent

Agent.register('OracleGuidedCodeActAgent', OracleGuidedCodeActAgent)
