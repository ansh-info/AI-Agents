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
        self.setup_workflow()

    def setup_workflow(self):
        """Setup the workflow graph with proper state transitions"""
        self.graph = StateGraph(AgentState)

        # Add nodes for different states
        self.graph.add_node("start", self.handle_start)
        self.graph.add_node("process", self.process_command)
        self.graph.add_node("success", self.handle_success)
        self.graph.add_node("error", self.handle_error)

        # Add edges to define transitions
        self.graph.add_edge("start", "process")
        self.graph.add_edge("process", "success")
        self.graph.add_edge("process", "error")
        self.graph.add_edge("success", END)
        self.graph.add_edge("error", END)

        # Set entry point
        self.graph.set_entry_point("start")

    def handle_start(self, state: AgentState) -> Dict[str, Any]:
        """Initialize processing of a new command"""
        state.update_status(AgentStatus.PROCESSING)
        state.current_step = "processing"
        return {"state": state, "next": "process"}

    def process_command(self, state: AgentState) -> Dict[str, Any]:
        """Process the command and determine next step"""
        try:
            # Get the last command from messages
            if state.memory.messages:
                last_message = state.memory.messages[-1]
                command = last_message["content"]

                # Here you can add different command processing logic
                if command.lower().startswith("search"):
                    # Will be implemented later for search functionality
                    state.current_step = "search_complete"
                elif command.lower().startswith("help"):
                    state.add_message("system", "Available commands: search, help")
                else:
                    state.add_message("system", f"Processed command: {command}")

                state.update_status(AgentStatus.SUCCESS)
                return {"state": state, "next": "success"}

            return {"state": state, "next": "success"}

        except Exception as e:
            state.update_status(AgentStatus.ERROR, str(e))
            return {"state": state, "next": "error"}

    def handle_success(self, state: AgentState) -> Dict[str, Any]:
        """Handle successful command processing"""
        if state.status != AgentStatus.SUCCESS:
            state.update_status(AgentStatus.SUCCESS)
        return {"state": state}

    def handle_error(self, state: AgentState) -> Dict[str, Any]:
        """Handle error states with proper error messages"""
        print(f"Error in workflow: {state.error_message}")
        return {"state": state}

    def process_command_external(self, command: str) -> AgentState:
        """External interface to process commands"""
        try:
            # Add the command to state memory
            self.current_state.add_message("user", command)

            # Run the workflow
            result = self.graph.run(self.current_state)

            # Update current state
            self.current_state = result

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
