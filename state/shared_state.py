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


def replace_list(existing: List[str], new: List[str]) -> List[str]:
    """Replace the existing list with the new one."""
    logger.info(f"Updating state list: {new}")
    return new


class Talk2Papers(MessagesState):
    """The state for the Talk2Papers agent."""

    papers: Annotated[list, replace_list]
    search_table: NotRequired[str]
    next: str  # Following LangGraph docs - required for routing
    current_agent: NotRequired[Optional[str]]
    is_last_step: Required[bool]  # Required field for LangGraph

    def log_state_update(self) -> None:
        """Log current state for debugging"""
        logger.info(
            f"Current State - Agent: {self.get('current_agent')}, Next: {self.get('next')}"
        )
        logger.info(f"Papers count: {len(self.get('papers', []))}")
