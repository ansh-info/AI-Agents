"""
This is the state file for the Talk2Papers agent.
"""

from typing import Annotated, List
import operator
from typing_extensions import TypedDict
from langgraph.prebuilt.chat_agent_executor import AgentState


def replace_list(existing: List[str], new: List[str]) -> List[str]:
    """Replace the existing list with the new one."""
    return new


class Talk2Papers(TypedDict, AgentState):
    """The state for the Talk2Papers agent."""

    papers: Annotated[list, replace_list] = []  # Replace instead of append
    search_table: str = ""  # For display
