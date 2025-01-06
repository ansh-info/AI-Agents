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
            system_prompt = """
            Analyze the user command and classify it into one of these categories:
            - SEARCH: Commands about searching or finding papers
            - HELP: Requests for help or information about commands
            - INVALID: Invalid or unknown commands
            Return ONLY the category name, nothing else.
            """

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

            updates = {
                "status": AgentStatus.SUCCESS,
                "current_step": "processing",
                "next_steps": ["finish"],
                "error_message": None,
                "memory": new_memory,
                "search_context": state.search_context,
            }

            if classification == "SEARCH":
                # Extract search query
                search_prompt = """
                Extract the search query from this command. Return ONLY the search terms,
                nothing else. If there's no clear search query, return 'INVALID'.
                """
                query = await self._get_llm_response(command, search_prompt)
                query = query.strip()

                if query and query.upper() != "INVALID":
                    new_memory.messages.append(
                        {"role": "system", "content": f"Searching for: {query}"}
                    )
                    updates.update(
                        {
                            "current_step": "search_initiated",
                            "search_context": SearchContext(query=query),
                        }
                    )
                else:
                    new_memory.messages.append(
                        {"role": "system", "content": "Invalid search query"}
                    )

            elif classification == "HELP":
                help_text = await self._get_llm_response(
                    "Generate a help message for the available commands",
                    "You are a helpful assistant explaining search commands. Keep it concise.",
                )
                new_memory.messages.append({"role": "system", "content": help_text})
                updates["current_step"] = "help_displayed"

            else:
                new_memory.messages.append(
                    {"role": "system", "content": f"Unknown command: {command}"}
                )
                updates["current_step"] = "command_processed"

            return updates

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
