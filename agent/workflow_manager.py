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

    def conditional_router(self, state: Dict) -> str:
        """Route based on state status"""
        if state["status"] == AgentStatus.ERROR:
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

    def handle_start(self, state: Dict) -> Dict[str, Any]:
        """Initialize the state for processing"""
        try:
            # Create new state modifications
            return {
                "status": AgentStatus.PROCESSING,
                "current_step": "started",
                "next_steps": ["process_command"],
            }
        except Exception as e:
            return {
                "status": AgentStatus.ERROR,
                "error_message": str(e),
                "current_step": "error",
                "next_steps": ["handle_error"],
            }

    def process_command(self, state: Dict) -> Dict[str, Any]:
        """Process the command based on its type"""
        try:
            last_message = state["memory"].messages[-1]["content"].lower()

            updates = {
                "current_step": "processing",
                "next_steps": ["finish"],
                "memory": {
                    "last_command": last_message,
                    "messages": state["memory"].messages.copy(),
                },
            }

            if last_message.startswith("search"):
                updates.update(
                    {
                        "status": AgentStatus.SUCCESS,
                        "current_step": "search_initiated",
                        "search_context": {"query": last_message[7:].strip()},
                    }
                )
                updates["memory"]["messages"].append(
                    {"role": "system", "content": "Search command received"}
                )

            elif last_message == "help":
                updates.update(
                    {"status": AgentStatus.SUCCESS, "current_step": "help_displayed"}
                )
                updates["memory"]["messages"].append(
                    {"role": "system", "content": "Available commands: search, help"}
                )

            else:
                updates.update(
                    {"status": AgentStatus.SUCCESS, "current_step": "command_processed"}
                )
                updates["memory"]["messages"].append(
                    {"role": "system", "content": f"Processed command: {last_message}"}
                )

            return updates

        except Exception as e:
            return {
                "status": AgentStatus.ERROR,
                "error_message": str(e),
                "current_step": "error",
                "next_steps": ["handle_error"],
            }

    def handle_error(self, state: Dict) -> Dict[str, Any]:
        """Handle error states"""
        return {
            "current_step": "error_handled",
            "next_steps": ["finish"],
            "error_message": state.get("error_message", "Unknown error occurred"),
            "status": AgentStatus.ERROR,
        }

    def handle_finish(self, state: Dict) -> Dict[str, Any]:
        """Finalize the command processing"""
        return {
            "current_step": "finished",
            "next_steps": [],
            "status": state.get("status", AgentStatus.SUCCESS),
        }

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
            result = self.graph.invoke(
                {
                    "state": self.current_state,
                    "status": self.current_state.status,
                    "memory": self.current_state.memory,
                    "current_step": self.current_state.current_step,
                    "error_message": self.current_state.error_message,
                    "search_context": self.current_state.search_context,
                    "next_steps": self.current_state.next_steps,
                }
            )

            # Update current state based on result
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
