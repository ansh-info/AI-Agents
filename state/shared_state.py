"""
This is the state file for the Talk2Papers agent.
"""

from typing import List
from typing_extensions import TypedDict
from langgraph.prebuilt.chat_agent_executor import AgentState


class PaperState(TypedDict):
    """State for managing papers."""

    papers: List[str]  # Current papers list
    search_table: str  # For display purposes


class Talk2Papers(AgentState, PaperState):
    """The state for the Talk2Papers agent."""

    pass
