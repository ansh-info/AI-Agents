"""
This is the state file for the Talk2Papers agent.
"""

from typing import Annotated, List
import operator
from typing_extensions import TypedDict
from langgraph.prebuilt.chat_agent_executor import AgentState


class Talk2Papers(TypedDict, AgentState):
    """The state for the Talk2Papers agent."""

    papers: Annotated[list, operator.add] = []  # Initialize as empty list
    search_table: str = ""  # Initialize with empty string
