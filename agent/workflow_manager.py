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

    def setup_workflow(self):
        """Setup the workflow graph with proper state transitions"""
        workflow = StateGraph(AgentState)

        # Define state transitions
        def conditional_edge(state: AgentState):
            if state.status == AgentStatus.ERROR:
                return "finish"
            return "process"

        # Add nodes
        workflow.add_node("start", self.handle_start)
        workflow.add_node("process", self.process_command)
        workflow.add_node("finish", self.handle_finish)

        # Add edges
        workflow.add_edge("start", conditional_edge)
        workflow.add_edge("process", "finish")
        workflow.add_edge("finish", END)

        # Set entry point
        workflow.set_entry_point("start")

        return workflow.compile()

    def handle_start(self, state: AgentState) -> Dict:
        """Initial state handling"""
        state_dict = {
            "status": AgentStatus.PROCESSING,
            "current_step": "started",
            "memory": {
                "messages": state.memory.messages,
                "current_context": state.memory.current_context,
                "last_command": state.memory.last_command,
            },
            "search_context": {
                "query": state.search_context.query,
                "current_page": state.search_context.current_page,
                "total_results": state.search_context.total_results,
            },
        }
        return {"state": state_dict, "next": "process"}

    def process_command(self, state: Dict) -> Dict:
        """Process the command"""
        try:
            messages = state["memory"]["messages"]
            if not messages:
                return {
                    "state": {
                        "status": AgentStatus.ERROR,
                        "error_message": "No command to process",
                        "current_step": "error",
                    }
                }

            command = messages[-1]["content"].lower()
            response_message = None

            if command.startswith("search"):
                response_message = "Search command received"
                current_step = "search_initiated"
                query = command[7:].strip()
                state["search_context"]["query"] = query
            elif command == "help":
                response_message = "Available commands: search, help"
                current_step = "help_displayed"
            else:
                response_message = f"Processed command: {command}"
                current_step = "command_processed"

            if response_message:
                messages.append({"role": "system", "content": response_message})

            return {
                "state": {
                    "status": AgentStatus.SUCCESS,
                    "current_step": current_step,
                    "memory": {
                        "messages": messages,
                        "current_context": state["memory"]["current_context"],
                        "last_command": command,
                    },
                    "search_context": state["search_context"],
                }
            }

        except Exception as e:
            return {
                "state": {
                    "status": AgentStatus.ERROR,
                    "error_message": str(e),
                    "current_step": "error",
                }
            }

    def handle_finish(self, state: Dict) -> Dict:
        """Finalize processing"""
        if "status" not in state:
            state["status"] = AgentStatus.SUCCESS
        return {"state": state}

    def process_command_external(self, command: str) -> AgentState:
        """External command processing interface"""
        try:
            # Prepare initial state
            self.current_state.add_message("user", command)

            # Convert state to dict for graph processing
            initial_state = {
                "status": self.current_state.status,
                "current_step": self.current_state.current_step,
                "memory": {
                    "messages": self.current_state.memory.messages,
                    "current_context": self.current_state.memory.current_context,
                    "last_command": self.current_state.memory.last_command,
                },
                "search_context": {
                    "query": self.current_state.search_context.query,
                    "current_page": self.current_state.search_context.current_page,
                    "total_results": self.current_state.search_context.total_results,
                },
            }

            # Run workflow
            result = self.graph.invoke({"state": initial_state})

            # Update current state from result
            state_dict = result["state"]

            # Create new AgentState instance
            new_state = AgentState()
            new_state.status = state_dict.get("status", AgentStatus.IDLE)
            new_state.current_step = state_dict.get("current_step", "initial")
            new_state.error_message = state_dict.get("error_message")

            if "memory" in state_dict:
                new_state.memory.messages = state_dict["memory"]["messages"]
                new_state.memory.current_context = state_dict["memory"][
                    "current_context"
                ]
                new_state.memory.last_command = state_dict["memory"]["last_command"]

            if "search_context" in state_dict:
                new_state.search_context.query = state_dict["search_context"]["query"]
                new_state.search_context.current_page = state_dict["search_context"][
                    "current_page"
                ]
                new_state.search_context.total_results = state_dict["search_context"][
                    "total_results"
                ]

            self.current_state = new_state
            return self.current_state

        except Exception as e:
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            return self.current_state

    def get_state(self) -> AgentState:
        """Get current state"""
        return self.current_state

    def reset_state(self):
        """Reset state"""
        self.current_state = AgentState()
