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
        """Initialize workflow manager"""
        self.current_state = AgentState()
        self.llm_client = OllamaClient()
        self.model_name = model_name

    async def process_command_async(self, command: str) -> AgentState:
        """Process command and update state asynchronously"""
        try:
            print(f"Debug: Processing command: {command}")

            # Reset state for new command
            self.reset_state()

            # Add command to state
            self.current_state.add_message("user", command)

            # Process command through steps
            await self._understand_command(self.current_state)
            await self._process_command(self.current_state)

            return self.current_state

        except Exception as e:
            print(f"Debug: Error in process_command_async: {str(e)}")
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            return self.current_state

    async def _understand_command(self, state: AgentState) -> None:
        """Understand and classify the command"""
        command = state.memory.messages[-1]["content"]

        # Simple rule-based classification instead of using LLM
        command_lower = command.lower().strip()

        if command_lower == "help":
            state.memory.current_context = "HELP"
            state.current_step = "help_displayed"
        elif command_lower.startswith("search"):
            state.memory.current_context = "SEARCH"
            state.current_step = "search_initiated"
        else:
            state.memory.current_context = "INVALID"
            state.current_step = "command_processed"

    async def _process_command(self, state: AgentState) -> None:
        """Process the classified command"""
        command = state.memory.messages[-1]["content"].lower()
        command_type = state.memory.current_context

        if command_type == "HELP":
            help_text = """Available commands:
1. search <query>: Search for academic papers (e.g., 'search LLM papers')
2. help: Show this help message

Example usage:
- search papers about machine learning
- search recent LLM papers
- help"""
            state.add_message("system", help_text)
            state.current_step = "help_displayed"

        elif command_type == "SEARCH":
            if command == "search":
                state.add_message(
                    "system",
                    "Please provide a search query. Example: 'search LLM papers'",
                )
                state.current_step = "invalid_search"
            else:
                # Extract query by removing 'search' prefix
                query = command.replace("search", "", 1).strip()
                state.add_message("system", f"Initiating search for: {query}")
                state.search_context.query = query
                state.current_step = "search_initiated"

        else:
            state.add_message("system", f"Unknown command: {command}")
            state.current_step = "command_processed"

        state.status = AgentStatus.SUCCESS

    def process_command_external(self, command: str) -> AgentState:
        """Synchronous interface for command processing"""
        return asyncio.run(self.process_command_async(command))

    def get_state(self) -> AgentState:
        """Get current state"""
        return self.current_state

    def reset_state(self):
        """Reset state to initial values"""
        self.current_state = AgentState()
