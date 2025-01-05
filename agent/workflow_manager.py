import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from copy import deepcopy
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from state.agent_state import (AgentState, AgentStatus, ConversationMemory,
                               SearchContext)


class WorkflowManager:
    def __init__(self):
        self.current_state = AgentState()
        self.graph = self.setup_workflow()

    def conditional_router(self, state: Dict[str, AgentState]) -> str:
        """Route based on state status"""
        current_state = state["state"]
        if current_state.status == AgentStatus.ERROR:
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
        current_state = state["state"]

        try:
            if not current_state.memory.messages:
                return {
                    "status": AgentStatus.ERROR,
                    "error_message": "No command to process",
                    "current_step": "error",
                }

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
            }

    def process_command(self, state: Dict[str, AgentState]) -> Dict[str, Any]:
        """Process the command based on its type"""
        current_state = state["state"]

        try:
            last_message = current_state.memory.messages[-1]["content"].lower()
            memory_update = ConversationMemory(
                messages=current_state.memory.messages.copy(), last_command=last_message
            )

            if last_message.startswith("search"):
                memory_update.messages.append(
                    {"role": "system", "content": "Search command received"}
                )
                search_context = SearchContext(query=last_message[7:].strip())

                return {
                    "status": AgentStatus.SUCCESS,
                    "current_step": "search_initiated",
                    "memory": memory_update,
                    "search_context": search_context,
                    "next_steps": ["finish"],
                }

            elif last_message == "help":
                memory_update.messages.append(
                    {"role": "system", "content": "Available commands: search, help"}
                )

                return {
                    "status": AgentStatus.SUCCESS,
                    "current_step": "help_displayed",
                    "memory": memory_update,
                    "next_steps": ["finish"],
                }

            else:
                memory_update.messages.append(
                    {"role": "system", "content": f"Processed command: {last_message}"}
                )

                return {
                    "status": AgentStatus.SUCCESS,
                    "current_step": "command_processed",
                    "memory": memory_update,
                    "next_steps": ["finish"],
                }

        except Exception as e:
            return {
                "status": AgentStatus.ERROR,
                "error_message": str(e),
                "current_step": "error",
                "next_steps": ["handle_error"],
            }

    def handle_error(self, state: Dict[str, AgentState]) -> Dict[str, Any]:
        """Handle error states"""
        return {
            "current_step": "error_handled",
            "next_steps": ["finish"],
            "status": AgentStatus.ERROR,
            "error_message": state["state"].error_message or "Unknown error occurred",
        }

    def handle_finish(self, state: Dict[str, AgentState]) -> Dict[str, Any]:
        """Finalize the command processing"""
        current_state = state["state"]
        return {
            "current_step": "finished",
            "next_steps": [],
            "status": (
                AgentStatus.SUCCESS
                if current_state.status != AgentStatus.ERROR
                else AgentStatus.ERROR
            ),
        }

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
