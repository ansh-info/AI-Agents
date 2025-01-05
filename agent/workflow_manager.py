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
        workflow.add_node("start", self._handle_start)
        workflow.add_node("process", self._process_command)
        workflow.add_node("finish", self._handle_finish)

        # Define edges
        workflow.add_edge("start", "process")
        workflow.add_edge("process", "finish")
        workflow.add_edge("finish", END)

        # Set entry point
        workflow.set_entry_point("start")

        return workflow.compile()

    def _handle_start(self, state: Dict[str, AgentState]) -> Dict[str, Any]:
        """Initialize the state for processing"""
        print("Debug: Entering handle_start")
        try:
            current_state = state["state"]

            # Create a new memory copy with the existing messages
            new_memory = ConversationMemory(
                messages=current_state.memory.messages.copy(), last_command=None
            )

            return {
                "status": AgentStatus.PROCESSING,
                "current_step": "processing",
                "next_steps": ["process"],
                "memory": new_memory,
            }
        except Exception as e:
            print(f"Debug: Error in handle_start: {str(e)}")
            return {
                "status": AgentStatus.ERROR,
                "error_message": str(e),
                "current_step": "error",
            }

    def _process_command(self, state: Dict[str, AgentState]) -> Dict[str, Any]:
        """Process the command based on its type"""
        print("Debug: Entering process_command")
        try:
            current_state = state["state"]
            command = current_state.memory.messages[-1]["content"].lower()

            # Create new memory with existing messages
            new_memory = ConversationMemory(
                messages=current_state.memory.messages.copy(), last_command=command
            )

            base_updates = {
                "status": AgentStatus.PROCESSING,
                "current_step": "processing",
                "next_steps": ["finish"],
                "memory": new_memory,
            }

            if command.startswith("search"):
                new_memory.messages.append(
                    {"role": "system", "content": "Search command received"}
                )
                base_updates.update(
                    {
                        "status": AgentStatus.SUCCESS,
                        "current_step": "search_initiated",
                        "search_context": SearchContext(query=command[7:].strip()),
                    }
                )

            elif command == "help":
                new_memory.messages.append(
                    {"role": "system", "content": "Available commands: search, help"}
                )
                base_updates.update(
                    {"status": AgentStatus.SUCCESS, "current_step": "help_displayed"}
                )

            else:
                new_memory.messages.append(
                    {"role": "system", "content": f"Processed command: {command}"}
                )
                base_updates.update(
                    {"status": AgentStatus.SUCCESS, "current_step": "command_processed"}
                )

            print(f"Debug: Returning updates from process_command: {base_updates}")
            return base_updates

        except Exception as e:
            print(f"Debug: Error in process_command: {str(e)}")
            return {
                "status": AgentStatus.ERROR,
                "error_message": str(e),
                "current_step": "error",
            }

    def _handle_finish(self, state: Dict[str, AgentState]) -> Dict[str, Any]:
        """Finalize the command processing"""
        print("Debug: Entering handle_finish")
        try:
            return {
                "status": AgentStatus.SUCCESS,
                "current_step": "finished",
                "next_steps": [],
            }
        except Exception as e:
            print(f"Debug: Error in handle_finish: {str(e)}")
            return {
                "status": AgentStatus.ERROR,
                "error_message": str(e),
                "current_step": "error",
            }

    def process_command_external(self, command: str) -> AgentState:
        """External interface to process commands"""
        try:
            print(f"Debug: Processing command: {command}")

            # Reset state for new command
            self.reset_state()
            print("Debug: State reset")

            # Add the command to state memory
            self.current_state.add_message("user", command)
            print("Debug: Message added to state")

            # Run the workflow
            print("Debug: About to invoke graph")
            result = self.graph.invoke(
                {
                    "state": self.current_state,
                }
            )
            print("Debug: Graph invoked, getting result")

            # Update current state
            self.current_state = result["state"]
            print("Debug: State updated")
            return self.current_state

        except Exception as e:
            print(f"Debug: Error in process_command_external: {str(e)}")
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            return self.current_state

    def get_state(self) -> AgentState:
        """Get current state"""
        return self.current_state

    def reset_state(self):
        """Reset the state to initial values"""
        self.current_state = AgentState()
