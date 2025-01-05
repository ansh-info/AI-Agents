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

    def setup_workflow(self):
        """Setup the workflow graph with proper state transitions"""
        workflow = StateGraph(AgentState)

        # Define the nodes
        workflow.add_node("start", self.handle_start)
        workflow.add_node("process", self.process_command)
        workflow.add_node("error", self.handle_error)
        workflow.add_node("finish", self.handle_finish)

        # Define edges
        workflow.add_edge("start", "process")
        workflow.add_edge("process", "finish")
        workflow.add_edge("error", "finish")
        workflow.add_edge("finish", END)

        # Set entry point
        workflow.set_entry_point("start")

        return workflow.compile()

    def handle_start(self, state: Dict) -> Dict[str, Any]:
        """Initialize the state for processing"""
        try:
            # Return updates dictionary
            return {
                "status": AgentStatus.PROCESSING,
                "current_step": "started",
                "next_steps": ["process"],
                "memory": state["state"].memory.model_copy(),
            }
        except Exception as e:
            return {
                "status": AgentStatus.ERROR,
                "error_message": str(e),
                "current_step": "error",
                "next_steps": ["error"],
            }

    def process_command(self, state: Dict) -> Dict[str, Any]:
        """Process the command based on its type"""
        try:
            current_state = state["state"]
            command = current_state.memory.messages[-1]["content"].lower()

            # Create memory update
            memory_update = current_state.memory.model_copy()
            memory_update.last_command = command

            updates = {
                "memory": memory_update,
                "current_step": "processing",
                "next_steps": ["finish"],
            }

            if command.startswith("search"):
                updates.update(
                    {
                        "status": AgentStatus.SUCCESS,
                        "current_step": "search_initiated",
                        "search_context": current_state.search_context.model_copy(
                            update={"query": command[7:].strip()}
                        ),
                    }
                )
                memory_update.messages.append(
                    {"role": "system", "content": "Search command received"}
                )

            elif command == "help":
                updates.update(
                    {
                        "status": AgentStatus.SUCCESS,
                        "current_step": "help_displayed",
                    }
                )
                memory_update.messages.append(
                    {"role": "system", "content": "Available commands: search, help"}
                )

            else:
                updates.update(
                    {
                        "status": AgentStatus.SUCCESS,
                        "current_step": "command_processed",
                    }
                )
                memory_update.messages.append(
                    {"role": "system", "content": f"Processed command: {command}"}
                )

            return updates

        except Exception as e:
            return {
                "status": AgentStatus.ERROR,
                "error_message": str(e),
                "current_step": "error",
                "next_steps": ["error"],
            }

    def handle_error(self, state: Dict) -> Dict[str, Any]:
        """Handle error states"""
        return {
            "status": AgentStatus.ERROR,
            "current_step": "error_handled",
            "next_steps": ["finish"],
            "error_message": state["state"].error_message or "Unknown error occurred",
        }

    def handle_finish(self, state: Dict) -> Dict[str, Any]:
        """Finalize the command processing"""
        return {
            "status": state["state"].status,
            "current_step": "finished",
            "next_steps": [],
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
