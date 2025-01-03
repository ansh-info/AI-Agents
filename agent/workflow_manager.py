import json
import os
import sys
from typing import Any, Dict

from langgraph.graph import END, StateGraph

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from copy import deepcopy
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from state.agent_state import AgentState, AgentStatus


class WorkflowManager:
    def __init__(self):
        self.current_state = AgentState()
        self.graph = self.setup_workflow()

    def conditional_router(self, state: Dict[str, AgentState]) -> str:
        """Route based on state status"""
        if state["state"].status == AgentStatus.ERROR:
            return "error"
        return "process"

    def setup_workflow(self):
        """Setup the workflow graph with proper state transitions"""
        workflow = StateGraph(AgentState)

        # Define the nodes
        workflow.add_node("start", self.handle_start)
        workflow.add_node("process", self.process_command)
        workflow.add_node("error", self.handle_error)
        workflow.add_node("finish", self.handle_finish)

        # Define edges with conditional routing
        workflow.add_conditional_edges(
            "start", self.conditional_router, {"error": "error", "process": "process"}
        )
        workflow.add_edge("process", "finish")
        workflow.add_edge("error", "finish")
        workflow.add_edge("finish", END)

        # Set entry point
        workflow.set_entry_point("start")

        return workflow.compile()

    def handle_start(self, state: Dict[str, AgentState]) -> Dict[str, Any]:
        """Initialize the state for processing"""
        try:
            current_state = state["state"]
            current_state.status = AgentStatus.PROCESSING
            current_state.current_step = "started"

            # Validate command presence
            if not current_state.memory.messages:
                current_state.status = AgentStatus.ERROR
                current_state.error_message = "No command to process"

            return {"state": current_state}
        except Exception as e:
            current_state = state["state"]
            current_state.status = AgentStatus.ERROR
            current_state.error_message = str(e)
            return {"state": current_state}

    def process_command(self, state: Dict[str, AgentState]) -> Dict[str, Any]:
        """Process the command based on its type"""
        try:
            current_state = state["state"]
            command = current_state.memory.messages[-1]["content"].lower()
            current_state.memory.last_command = command
            current_state.current_step = "processing"

            if command.startswith("search"):
                current_state.add_message("system", "Search command received")
                current_state.current_step = "search_initiated"
                current_state.search_context.query = command[7:].strip()
                current_state.status = AgentStatus.SUCCESS

            elif command == "help":
                current_state.add_message("system", "Available commands: search, help")
                current_state.current_step = "help_displayed"
                current_state.status = AgentStatus.SUCCESS

            else:
                current_state.add_message("system", f"Processed command: {command}")
                current_state.current_step = "command_processed"
                current_state.status = AgentStatus.SUCCESS

            return {"state": current_state}

        except Exception as e:
            current_state = state["state"]
            current_state.status = AgentStatus.ERROR
            current_state.error_message = str(e)
            return {"state": current_state}

    def handle_error(self, state: Dict[str, AgentState]) -> Dict[str, Any]:
        """Handle error states"""
        current_state = state["state"]
        if current_state.error_message is None:
            current_state.error_message = "Unknown error occurred"
        current_state.current_step = "error_handled"
        return {"state": current_state}

    def handle_finish(self, state: Dict[str, AgentState]) -> Dict[str, Any]:
        """Finalize the command processing"""
        current_state = state["state"]
        if current_state.status != AgentStatus.ERROR:
            current_state.status = AgentStatus.SUCCESS
        current_state.current_step = "finished"
        return {"state": current_state}

    def process_command_external(self, command: str) -> AgentState:
        """External interface to process commands"""
        try:
            # Reset state for new command
            self.reset_state()

            # Add the command to state memory
            self.current_state.add_message("user", command)

            # Run the workflow
            result = self.graph.invoke({"state": self.current_state})

            # Update current state
            self.current_state = result["state"]
            return self.current_state

        except Exception as e:
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            return self.current_state

    def get_state(self) -> AgentState:
        """Get current state"""
        return self.current_state

    def reset_state(self):
        """Reset the state to initial values"""
        self.current_state = AgentState()
