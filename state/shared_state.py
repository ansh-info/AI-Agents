"""
This is the state file for the Talk2Papers agent.
"""

import logging
from typing import Annotated, List, Optional

from langgraph.prebuilt import AgentState
from typing_extensions import TypedDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def replace_list(existing: List[str], new: List[str]) -> List[str]:
    """Replace the existing list with the new one."""
    logger.info(f"Updating state list: {new}")
    return new


class Talk2Papers(TypedDict, AgentState):
    """The state for the Talk2Papers agent."""

    papers: Annotated[list, replace_list] = []  # Replace instead of append
    search_table: str = ""  # For display
    next: Optional[str] = None  # For routing between agents
    current_agent: Optional[str] = None  # Track current active agent

    def log_state_update(self) -> None:
        """Log current state for debugging"""
        logger.info(
            f"Current State - Agent: {self.get('current_agent')}, Next: {self.get('next')}"
        )
        logger.info(f"Papers count: {len(self.get('papers', []))}")
