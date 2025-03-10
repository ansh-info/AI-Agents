"""
This is the state file for the Talk2Papers agent.
"""

from typing import List, Annotated
from langgraph.prebuilt.chat_agent_executor import AgentState


class Talk2Papers(AgentState):
    """The state for the Talk2Papers agent."""

    search_table: str
    papers: Annotated[List[str], "concurrent"]
