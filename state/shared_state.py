"""
This is the state file for the Talk2Papers agent.
"""

import logging
from typing import Annotated, List, Optional, NotRequired, TypedDict

from typing_extensions import Required

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def replace_list(existing: List[str], new: List[str]) -> List[str]:
    """Replace the existing list with the new one."""
    logger.info(f"Updating state list: {new}")
    return new


class Talk2Papers(TypedDict, total=False):
    """The state for the Talk2Papers agent."""

    papers: Annotated[list, replace_list]
    search_table: NotRequired[str]
    next: NotRequired[Optional[str]]
    current_agent: NotRequired[Optional[str]]
    messages: Required[List[dict]]
    is_last_step: Required[bool]  # Required field for LangGraph

    def log_state_update(self) -> None:
        """Log current state for debugging"""
        logger.info(
            f"Current State - Agent: {self.get('current_agent')}, Next: {self.get('next')}"
        )
        logger.info(f"Papers count: {len(self.get('papers', []))}")
