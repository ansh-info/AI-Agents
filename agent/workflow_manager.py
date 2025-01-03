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

    def setup_workflow(self):
        """Setup the workflow graph with proper state transitions"""
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

        return workflow.compile()

    def handle_start(self, state: AgentState) -> Dict[str, Any]:
        """Initialize the state for processing"""
        new_state = deepcopy(state)
        new_state.status = AgentStatus.PROCESSING
        new_state.current_step = "started"
        return {"state": new_state, "next": "route"}

    def route_command(self, state: AgentState) -> Dict[str, Any]:
        """Route the command to appropriate handler"""
        new_state = deepcopy(state)

        if not new_state.memory.messages:
            new_state.status = AgentStatus.ERROR
            new_state.error_message = "No command to process"
            return {"state": new_state, "next": "finish"}

        command = new_state.memory.messages[-1]["content"].lower()
        new_state.memory.last_command = command
        return {"state": new_state, "next": "process"}

    def process_command(self, state: AgentState) -> Dict[str, Any]:
        """Process the command based on its type"""
        try:
            new_state = deepcopy(state)
            command = new_state.memory.last_command

            if command.startswith("search"):
                new_state.add_message("system", "Search command received")
                new_state.current_step = "search_initiated"
                new_state.search_context.query = command[
                    7:
                ].strip()  # Remove "search " prefix

            elif command == "help":
                new_state.add_message("system", "Available commands: search, help")
                new_state.current_step = "help_displayed"

            else:
                new_state.add_message("system", f"Processed command: {command}")
                new_state.current_step = "command_processed"

            new_state.status = AgentStatus.SUCCESS
            return {"state": new_state, "next": "finish"}

        except Exception as e:
            new_state = deepcopy(state)
            new_state.status = AgentStatus.ERROR
            new_state.error_message = str(e)
            return {"state": new_state, "next": "finish"}

    def handle_finish(self, state: AgentState) -> Dict[str, Any]:
        """Finalize the command processing"""
        new_state = deepcopy(state)
        if new_state.status != AgentStatus.ERROR:
            new_state.status = AgentStatus.SUCCESS
        return {"state": new_state}

    def process_command_external(self, command: str) -> AgentState:
        """External interface to process commands"""
        try:
            # Create new state for processing
            new_state = deepcopy(self.current_state)
            new_state.add_message("user", command)

            # Run the workflow
            config = {"state": new_state}
            result = self.graph.invoke(config)

            # Update current state
            self.current_state = result["state"]
            return self.current_state

        except Exception as e:
            new_state = deepcopy(self.current_state)
            new_state.status = AgentStatus.ERROR
            new_state.error_message = str(e)
            self.current_state = new_state
            return self.current_state

    def get_state(self) -> AgentState:
        """Get current state"""
        return self.current_state

    def reset_state(self):
        """Reset the state to initial values"""
        self.current_state = AgentState()
