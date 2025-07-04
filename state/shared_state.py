"""
This is the state file for the Talk2Papers agent.
"""

import logging
from typing import Annotated, List, NotRequired, Optional
from typing_extensions import Required

from langgraph.graph import MessagesState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from typing import Dict, Optional, Any
from typing_extensions import NotRequired, Required


def replace_dict(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Replace the existing dict with the new one."""
    logger.info("Updating existing state %s with the state dict: %s", existing, new)
    return new


class Talk2Competitors(AgentState):
    """
    The state for the talk2comp agent, inheriting from AgentState.
    """

    papers: Annotated[Dict[str, Any], replace_dict]  # Changed from List to Dict
    search_table: NotRequired[str]
    next: str  # Required for routing in LangGraph
    current_agent: NotRequired[Optional[str]]
    is_last_step: Required[bool]  # Required field for LangGraph
    llm_model: str
