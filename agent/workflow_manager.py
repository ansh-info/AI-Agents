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

    def _handle_start(self, state: AgentState) -> Dict[str, Any]:
        """Initialize the state for processing"""
        print("Debug: Entering handle_start")
        try:
            command = (
                state.memory.messages[-1]["content"] if state.memory.messages else ""
            )

            updates = {
                "status": AgentStatus.PROCESSING,
                "current_step": "processing",
                "next_steps": ["process"],
                "error_message": None,
                "memory": ConversationMemory(
                    messages=state.memory.messages.copy(), last_command=command
                ),
                "search_context": SearchContext(),
            }
            print("Debug: Start node returning updates", updates["status"])
            return updates

        except Exception as e:
            print(f"Debug: Error in handle_start: {str(e)}")
            return {
                "status": AgentStatus.ERROR,
                "error_message": str(e),
                "current_step": "error",
                "next_steps": ["finish"],
                "memory": ConversationMemory(),
                "search_context": SearchContext(),
            }

    def _process_command(self, state: AgentState) -> Dict[str, Any]:
        """Process the command based on its type"""
        print("Debug: Entering process_command")
        try:
            command = state.memory.messages[-1]["content"].lower()

            # Create base memory and updates
            new_memory = ConversationMemory(
                messages=state.memory.messages.copy(), last_command=command
            )

            updates = {
                "status": AgentStatus.SUCCESS,
                "current_step": "processing",
                "next_steps": ["finish"],
                "error_message": None,
                "memory": new_memory,
                "search_context": state.search_context,
            }

            if command.startswith("search"):
                new_memory.messages.append(
                    {"role": "system", "content": "Search command received"}
                )
                updates.update(
                    {
                        "current_step": "search_initiated",
                        "search_context": SearchContext(query=command[7:].strip()),
                    }
                )

            elif command == "help":
                new_memory.messages.append(
                    {"role": "system", "content": "Available commands: search, help"}
                )
                updates["current_step"] = "help_displayed"

            else:
                new_memory.messages.append(
                    {"role": "system", "content": f"Processed command: {command}"}
                )
                updates["current_step"] = "command_processed"

            print(f"Debug: Process command returning updates")
            return updates

        except Exception as e:
            print(f"Debug: Error in process_command: {str(e)}")
            return {
                "status": AgentStatus.ERROR,
                "error_message": str(e),
                "current_step": "error",
                "next_steps": ["finish"],
                "memory": ConversationMemory(),
                "search_context": SearchContext(),
            }

    def _handle_finish(self, state: AgentState) -> Dict[str, Any]:
        """Finalize the command processing"""
        print("Debug: Entering handle_finish")
        try:
            return {
                "status": AgentStatus.SUCCESS,
                "current_step": "finished",
                "next_steps": [],
                "error_message": None,
                "memory": state.memory,
                "search_context": state.search_context,
            }
        except Exception as e:
            print(f"Debug: Error in handle_finish: {str(e)}")
            return {
                "status": AgentStatus.ERROR,
                "error_message": str(e),
                "current_step": "error",
                "next_steps": [],
                "memory": ConversationMemory(),
                "search_context": SearchContext(),
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

            print("Debug: About to invoke graph")
            result_dict = self.graph.invoke(self.current_state)
            print("Debug: Graph invoked, getting result")

            # Convert result dictionary back to AgentState
            print("Debug: Converting result to AgentState")
            final_state = AgentState(
                status=result_dict["status"],
                error_message=result_dict.get("error_message"),
                search_context=result_dict.get("search_context", SearchContext()),
                memory=result_dict.get("memory", ConversationMemory()),
                current_step=result_dict.get("current_step", "finished"),
                next_steps=result_dict.get("next_steps", []),
            )
            print("Debug: State converted")
            return final_state

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
