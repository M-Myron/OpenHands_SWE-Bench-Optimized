from openhands.agenthub.oracle_guided_v2_codeact_agent.oracle_guided_v2_codeact_agent import (
    OracleGuidedV2CodeActAgent,
)
from openhands.controller.agent import Agent

Agent.register('OracleGuidedV2CodeActAgent', OracleGuidedV2CodeActAgent)
