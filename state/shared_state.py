"""
This is the state file for the Talk2Papers agent.
"""

from typing import Annotated
import operator
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing_extensions import TypedDict


class Talk2Papers(TypedDict, AgentState):
    """The state for the Talk2Papers agent."""

    papers: Annotated[list, operator.add]
    search_table: str
