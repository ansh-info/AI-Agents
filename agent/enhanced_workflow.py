import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from clients.ollama_client import OllamaClient
from state.agent_state import (AgentState, AgentStatus, ConversationMemory,
                               SearchContext)


class EnhancedWorkflowManager:
    def __init__(self, model_name: str = "llama3.2:1b"):
        self.current_state = AgentState()
        self.llm_client = OllamaClient()
        self.model_name = model_name
        self.graph = self.setup_workflow()

    def setup_workflow(self) -> StateGraph:
        """Setup the workflow graph with LLM capabilities"""
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("start", self._handle_start)
        workflow.add_node("understand", self._understand_command)
        workflow.add_node("process", self._process_command)
        workflow.add_node("finish", self._handle_finish)

        # Define edges
        workflow.add_edge("start", "understand")
        workflow.add_edge("understand", "process")
        workflow.add_edge("process", "finish")
        workflow.add_edge("finish", END)

        workflow.set_entry_point("start")
        return workflow.compile()

    async def _get_llm_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Get response from LLM"""
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                model=self.model_name,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            return response
        except Exception as e:
            raise Exception(f"LLM error: {str(e)}")

    def _handle_start(self, state: AgentState) -> Dict[str, Any]:
        """Initialize the state for processing"""
        try:
            command = (
                state.memory.messages[-1]["content"] if state.memory.messages else ""
            )

            return {
                "status": AgentStatus.PROCESSING,
                "current_step": "understanding",
                "next_steps": ["understand"],
                "error_message": None,
                "memory": ConversationMemory(
                    messages=state.memory.messages.copy(), last_command=command
                ),
                "search_context": SearchContext(),
            }
        except Exception as e:
            return self._create_error_state(str(e))

    async def _understand_command(self, state: AgentState) -> Dict[str, Any]:
        """Use LLM to understand the command"""
        try:
            command = state.memory.messages[-1]["content"]

            # Classify command using LLM
            system_prompt = """You are a command classifier. Analyze the command and respond with exactly one word from: SEARCH, HELP, or INVALID.
            Rules:
            - SEARCH: If command contains 'search' or asks to find/look for papers
            - HELP: If command is 'help' or asks for assistance/commands
            - INVALID: For any other command
            Respond with ONLY the classification word."""

            classification = await self._get_llm_response(command, system_prompt)
            classification = classification.strip().upper()

            new_memory = ConversationMemory(
                messages=state.memory.messages.copy(),
                last_command=command,
                current_context=classification,
            )

            return {
                "status": AgentStatus.PROCESSING,
                "current_step": "processing",
                "next_steps": ["process"],
                "error_message": None,
                "memory": new_memory,
                "search_context": state.search_context,
            }
        except Exception as e:
            return self._create_error_state(str(e))

    async def _process_command(self, state: AgentState) -> Dict[str, Any]:
        """Process the command based on LLM classification"""
        try:
            command = state.memory.messages[-1]["content"].lower()
            classification = state.memory.current_context

            new_memory = ConversationMemory(
                messages=state.memory.messages.copy(),
                last_command=command,
                current_context=classification,
            )

            # Base state updates
            base_updates = {
                "status": AgentStatus.SUCCESS,
                "error_message": None,
                "memory": new_memory,
                "search_context": state.search_context,
            }

            if classification == "SEARCH":
                # Extract search query - remove 'search' from the beginning if present
                query = command.lower().replace("search", "", 1).strip()

                if query:
                    new_memory.messages.append(
                        {"role": "system", "content": f"Initiating search for: {query}"}
                    )
                    return {
                        **base_updates,
                        "current_step": "search_initiated",
                        "next_steps": ["finish"],
                        "search_context": SearchContext(query=query),
                    }
                else:
                    new_memory.messages.append(
                        {
                            "role": "system",
                            "content": "Please provide a search query. Example: 'search LLM papers'",
                        }
                    )
                    return {
                        **base_updates,
                        "current_step": "invalid_search",
                        "next_steps": ["finish"],
                    }

            elif classification == "HELP":
                help_text = """Available commands:
1. search <query>: Search for academic papers (e.g., 'search LLM papers')
2. help: Show this help message

Example usage:
- search papers about machine learning
- search recent LLM papers
- help"""
                new_memory.messages.append({"role": "system", "content": help_text})
                return {
                    **base_updates,
                    "current_step": "help_displayed",
                    "next_steps": ["finish"],
                }

            else:
                new_memory.messages.append(
                    {"role": "system", "content": f"Unknown command: {command}"}
                )
                return {
                    **base_updates,
                    "current_step": "command_processed",
                    "next_steps": ["finish"],
                }

        except Exception as e:
            return self._create_error_state(str(e))

    def _handle_finish(self, state: AgentState) -> Dict[str, Any]:
        """Finalize the command processing"""
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
            return self._create_error_state(str(e))

    def _create_error_state(self, error_message: str) -> Dict[str, Any]:
        """Helper method to create error state"""
        return {
            "status": AgentStatus.ERROR,
            "error_message": error_message,
            "current_step": "error",
            "next_steps": ["finish"],
            "memory": ConversationMemory(),
            "search_context": SearchContext(),
        }

    async def process_command_async(self, command: str) -> AgentState:
        """Async interface to process commands"""
        try:
            print(f"Debug: Processing command: {command}")

            # Reset state for new command
            self.reset_state()

            # Add command to state memory
            self.current_state.add_message("user", command)

            # Process through workflow
            result_dict = await self.graph.ainvoke(self.current_state)

            # Convert result dictionary back to AgentState
            final_state = AgentState(
                status=result_dict["status"],
                error_message=result_dict.get("error_message"),
                search_context=result_dict.get("search_context", SearchContext()),
                memory=result_dict.get("memory", ConversationMemory()),
                current_step=result_dict.get("current_step", "finished"),
                next_steps=result_dict.get("next_steps", []),
            )
            return final_state

        except Exception as e:
            print(f"Debug: Error in process_command_async: {str(e)}")
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            return self.current_state

    def process_command_external(self, command: str) -> AgentState:
        """Synchronous interface to process commands"""
        return asyncio.run(self.process_command_async(command))

    def get_state(self) -> AgentState:
        """Get current state"""
        return self.current_state

    def reset_state(self):
        """Reset the state to initial values"""
        self.current_state = AgentState()
