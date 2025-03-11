"""
This is the state file for the Talk2Papers agent.
"""

from typing import Annotated, List, TypeVar
from operator import add
from langgraph.prebuilt.chat_agent_executor import AgentState

T = TypeVar("T")


class Talk2Papers(AgentState):
    """The state for the Talk2Papers agent."""

    # Using operator.add as reducer to make this append-only
    papers: Annotated[List[str], add] = []  # Initialize as empty list
    search_table: str = ""  # Keep this as is since it doesn't need concurrent updates
