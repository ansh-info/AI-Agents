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

    def handle_start(self, state: AgentState) -> Dict[str, Any]:
        """Initialize the state for processing"""
        try:
            state.status = AgentStatus.PROCESSING
            state.current_step = "started"

            # Validate command presence
            if not state.memory.messages:
                state.status = AgentStatus.ERROR
                state.error_message = "No command to process"

            return {"state": state}
        except Exception as e:
            state.status = AgentStatus.ERROR
            state.error_message = str(e)
            return {"state": state}

    def process_command(self, state: AgentState) -> Dict[str, Any]:
        """Process the command based on its type"""
        try:
            command = state.memory.messages[-1]["content"].lower()
            state.memory.last_command = command

            if command.startswith("search"):
                state.add_message("system", "Search command received")
                state.current_step = "search_initiated"
                state.search_context.query = command[7:].strip()

            elif command == "help":
                state.add_message("system", "Available commands: search, help")
                state.current_step = "help_displayed"

            else:
                state.add_message("system", f"Processed command: {command}")
                state.current_step = "command_processed"

            state.status = AgentStatus.SUCCESS
            return {"state": state}

        except Exception as e:
            state.status = AgentStatus.ERROR
            state.error_message = str(e)
            return {"state": state}

    def handle_error(self, state: AgentState) -> Dict[str, Any]:
        """Handle error states"""
        if state.error_message is None:
            state.error_message = "Unknown error occurred"
        return {"state": state}

    def handle_finish(self, state: AgentState) -> Dict[str, Any]:
        """Finalize the command processing"""
        if state.status != AgentStatus.ERROR:
            state.status = AgentStatus.SUCCESS
        return {"state": state}

    def process_command_external(self, command: str) -> AgentState:
        """External interface to process commands"""
        try:
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
