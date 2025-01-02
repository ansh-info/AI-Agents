from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field


# Configure Pydantic model behavior
class ConfigDict:
    arbitrary_types_allowed = True


class AgentStatus(Enum):
    """Status of the current agent operation"""

    IDLE = "idle"
    PROCESSING = "processing"
    SUCCESS = "success"
    ERROR = "error"


class SearchContext(BaseModel):
    """Context for search operations"""

    query: str = ""
    results: Optional[Any] = None  # Changed from DataFrame to Any
    current_page: int = 1
    total_results: int = 0
    selected_paper_index: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types like DataFrame


class ConversationMemory(BaseModel):
    """Track conversation history and context"""

    messages: List[Dict[str, str]] = Field(default_factory=list)
    current_context: Optional[str] = None
    last_command: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class AgentState(BaseModel):
    """Main state container for the agent workflow"""

    # Agent status tracking
    status: AgentStatus = AgentStatus.IDLE
    error_message: Optional[str] = None

    # Search context
    search_context: SearchContext = Field(default_factory=SearchContext)

    # Conversation memory
    memory: ConversationMemory = Field(default_factory=ConversationMemory)

    # Workflow state
    current_step: str = "initial"
    next_steps: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def update_status(self, new_status: AgentStatus, error: Optional[str] = None):
        """Update agent status and error message"""
        self.status = new_status
        self.error_message = error if new_status == AgentStatus.ERROR else None

    def add_message(self, role: str, content: str):
        """Add a message to conversation memory"""
        self.memory.messages.append({"role": role, "content": content})

    def update_search_results(self, results: pd.DataFrame, total_results: int):
        """Update search results in context"""
        self.search_context.results = results
        self.search_context.total_results = total_results
        self.update_status(AgentStatus.SUCCESS)

    def get_current_paper(self) -> Optional[Dict[str, Any]]:
        """Get currently selected paper details"""
        if (
            self.search_context.results is not None
            and self.search_context.selected_paper_index is not None
        ):
            return self.search_context.results.iloc[
                self.search_context.selected_paper_index
            ].to_dict()
        return None


def create_workflow_manager():
    """Create and configure the workflow manager"""
    # Initialize state graph
    workflow = StateGraph(AgentState)

    # Define state transitions
    def handle_initial_state(state: AgentState) -> Dict:
        """Handle initial state and determine next step"""
        # Reset state for new operation
        state.update_status(AgentStatus.IDLE)
        state.current_step = "ready"
        return {"state": state}

    def handle_error(state: AgentState) -> Dict:
        """Handle error states"""
        state.update_status(AgentStatus.ERROR)
        return {"state": state}

    # Add nodes to workflow
    workflow.add_node("initial", handle_initial_state)
    workflow.add_node("error", handle_error)

    # Set entry point
    workflow.set_entry_point("initial")

    return workflow


class WorkflowManager:
    """Manager class for workflow operations"""

    def __init__(self):
        self.workflow = create_workflow_manager()
        self.current_state = AgentState()

    def process_command(self, command: str) -> AgentState:
        """Process a new command and update state"""
        try:
            # Add command to memory
            self.current_state.add_message("user", command)

            # Run workflow
            self.current_state = self.workflow.run(self.current_state)

            return self.current_state
        except Exception as e:
            self.current_state.update_status(AgentStatus.ERROR, str(e))
            return self.current_state

    def get_state(self) -> AgentState:
        """Get current state"""
        return self.current_state
