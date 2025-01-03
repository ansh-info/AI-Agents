import os
import sys
from typing import Dict

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from langgraph.graph import END, StateGraph

from state.agent_state import AgentState, AgentStatus


class WorkflowManager:
    def __init__(self):
        self.current_state = AgentState()
        self.setup_workflow()

    def setup_workflow(self):
        self.graph = StateGraph(AgentState)

        # Add nodes
        self.graph.add_node("process", self.process_state)
        self.graph.add_node("error", self.handle_error)

        # Add edges
        self.graph.add_edge("process", "error")
        self.graph.add_edge("error", END)

        # Set entry point
        self.graph.set_entry_point("process")

    def process_state(self, state: AgentState) -> Dict:
        """Process the current state"""
        try:
            state.update_status(AgentStatus.PROCESSING)
            # Add any processing logic here
            state.update_status(AgentStatus.SUCCESS)
            return {"state": state}
        except Exception as e:
            state.update_status(AgentStatus.ERROR, str(e))
            return {"state": state, "next": "error"}

    def handle_error(self, state: AgentState) -> Dict:
        """Handle error states"""
        print(f"Handling error: {state.error_message}")
        return {"state": state}

    def process_command(self, command: str) -> AgentState:
        """Process a new command and update state"""
        try:
            self.current_state.add_message("user", command)
            result = self.graph.run(self.current_state)
            self.current_state = result
            return self.current_state
        except Exception as e:
            self.current_state.update_status(AgentStatus.ERROR, str(e))
            return self.current_state

    def get_state(self) -> AgentState:
        return self.current_state
