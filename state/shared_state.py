"""
This is the state file for the Talk2Papers agent.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from langgraph.graph import START, StateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState

from config.config import config


class Talk2Papers(AgentState):
    """
    The state for the Talk2Papers agent.
    """

    search_table: str
    papers: List
