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

    def handle_start(self, state: Dict[str, AgentState]) -> Dict[str, Dict[str, Any]]:
        """Initialize the state for processing"""
        current_state = state["state"]
        updates = {}

        try:
            updates["status"] = AgentStatus.PROCESSING
            updates["current_step"] = "started"

            if not current_state.memory.messages:
                updates["status"] = AgentStatus.ERROR
                updates["error_message"] = "No command to process"

            # Create new state with updates
            new_state = current_state.model_copy(update=updates)
            return {"state": new_state}

        except Exception as e:
            updates["status"] = AgentStatus.ERROR
            updates["error_message"] = str(e)
            new_state = current_state.model_copy(update=updates)
            return {"state": new_state}

    def process_command(
        self, state: Dict[str, AgentState]
    ) -> Dict[str, Dict[str, Any]]:
        """Process the command based on its type"""
        current_state = state["state"]
        updates = {}

        try:
            command = current_state.memory.messages[-1]["content"].lower()

            # Common updates
            updates["memory"] = current_state.memory.model_copy()
            updates["memory"].last_command = command
            updates["current_step"] = "processing"

            if command.startswith("search"):
                updates["search_context"] = current_state.search_context.model_copy()
                updates["search_context"].query = command[7:].strip()
                updates["current_step"] = "search_initiated"
                updates["status"] = AgentStatus.SUCCESS
                new_memory = updates["memory"]
                new_memory.messages = current_state.memory.messages + [
                    {"role": "system", "content": "Search command received"}
                ]
                updates["memory"] = new_memory

            elif command == "help":
                updates["current_step"] = "help_displayed"
                updates["status"] = AgentStatus.SUCCESS
                new_memory = updates["memory"]
                new_memory.messages = current_state.memory.messages + [
                    {"role": "system", "content": "Available commands: search, help"}
                ]
                updates["memory"] = new_memory

            else:
                updates["current_step"] = "command_processed"
                updates["status"] = AgentStatus.SUCCESS
                new_memory = updates["memory"]
                new_memory.messages = current_state.memory.messages + [
                    {"role": "system", "content": f"Processed command: {command}"}
                ]
                updates["memory"] = new_memory

            new_state = current_state.model_copy(update=updates)
            return {"state": new_state}

        except Exception as e:
            updates["status"] = AgentStatus.ERROR
            updates["error_message"] = str(e)
            new_state = current_state.model_copy(update=updates)
            return {"state": new_state}

    def handle_error(self, state: Dict[str, AgentState]) -> Dict[str, Dict[str, Any]]:
        """Handle error states"""
        current_state = state["state"]
        updates = {
            "current_step": "error_handled",
        }

        if current_state.error_message is None:
            updates["error_message"] = "Unknown error occurred"

        new_state = current_state.model_copy(update=updates)
        return {"state": new_state}

    def handle_finish(self, state: Dict[str, AgentState]) -> Dict[str, Dict[str, Any]]:
        """Finalize the command processing"""
        current_state = state["state"]
        updates = {"current_step": "finished"}

        if current_state.status != AgentStatus.ERROR:
            updates["status"] = AgentStatus.SUCCESS

        new_state = current_state.model_copy(update=updates)
        return {"state": new_state}

    def process_command_external(self, command: str) -> AgentState:
        """External interface to process commands"""
        try:
            # Reset state for new command
            self.reset_state()

            # Add the command to state memory
            self.current_state.memory.messages.append(
                {"role": "user", "content": command}
            )

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
