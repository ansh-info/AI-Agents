import os
import sys
from typing import Any, Dict

from langgraph.graph import END, StateGraph

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from langgraph.graph import END, StateGraph

from state.agent_state import AgentState, AgentStatus


class WorkflowManager:
    def __init__(self):
        self.current_state = AgentState()
        self.graph = self.setup_workflow()

    def setup_workflow(self):
        """Setup the workflow graph with proper state transitions"""
        # Initialize workflow
        workflow = StateGraph(AgentState)

        # Define the nodes
        workflow.add_node("start", self.handle_start)
        workflow.add_node("route", self.route_command)
        workflow.add_node("process", self.process_command)
        workflow.add_node("finish", self.handle_finish)

        # Define the edges
        workflow.add_edge("start", "route")
        workflow.add_edge("route", "process")
        workflow.add_edge("process", "finish")
        workflow.add_edge("finish", END)

        # Set entry point
        workflow.set_entry_point("start")

        # Compile the graph
        return workflow.compile()

    def handle_start(self, state: AgentState) -> Dict[str, Any]:
        """Initialize the state for processing"""
        state.update_status(AgentStatus.PROCESSING)
        state.current_step = "started"
        return {"state": state, "next": "route"}

    def route_command(self, state: AgentState) -> Dict[str, Any]:
        """Route the command to appropriate handler"""
        if not state.memory.messages:
            state.update_status(AgentStatus.ERROR, "No command to process")
            return {"state": state, "next": "finish"}

        command = state.memory.messages[-1]["content"].lower()
        state.memory.last_command = command
        return {"state": state, "next": "process"}

    def process_command(self, state: AgentState) -> Dict[str, Any]:
        """Process the command based on its type"""
        try:
            command = state.memory.last_command

            if command.startswith("search"):
                # Handle search command
                state.add_message("system", "Search command received")
                state.current_step = "search_initiated"

            elif command == "help":
                # Handle help command
                state.add_message("system", "Available commands: search, help")
                state.current_step = "help_displayed"

            else:
                # Handle general command
                state.add_message("system", f"Processed command: {command}")
                state.current_step = "command_processed"

            state.update_status(AgentStatus.SUCCESS)
            return {"state": state, "next": "finish"}

        except Exception as e:
            state.update_status(AgentStatus.ERROR, str(e))
            return {"state": state, "next": "finish"}

    def handle_finish(self, state: AgentState) -> Dict[str, Any]:
        """Finalize the command processing"""
        if state.status != AgentStatus.ERROR:
            state.update_status(AgentStatus.SUCCESS)
        return {"state": state}

    def process_command_external(self, command: str) -> AgentState:
        """External interface to process commands"""
        try:
            # Add the command to state memory
            self.current_state.add_message("user", command)

            # Run the workflow with the current state
            config = {"state": self.current_state}
            result = self.graph.invoke(config)

            # Update current state
            self.current_state = result["state"]

            return self.current_state

        except Exception as e:
            self.current_state.update_status(AgentStatus.ERROR, str(e))
            return self.current_state

    def get_state(self) -> AgentState:
        """Get current state"""
        return self.current_state

    def reset_state(self):
        """Reset the state to initial values"""
        self.current_state = AgentState()
