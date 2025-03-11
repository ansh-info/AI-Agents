"""
This is the state file for the Talk2Papers agent.
"""

from typing import Annotated
import operator
from langgraph.prebuilt.chat_agent_executor import AgentState


class Talk2Papers(AgentState):
    """The state for the Talk2Papers agent."""

    papers: Annotated[
        list, operator.add
    ]  # Using list instead of List for concurrent updates
    search_table: str
