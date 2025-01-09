import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from clients.ollama_client import OllamaClient
from clients.ollama_enhanced import EnhancedOllamaClient
from clients.semantic_scholar_client import (SearchResults,
                                             SemanticScholarClient)
from state.agent_state import (AgentState, AgentStatus, ConversationMemory,
                               SearchContext)


class EnhancedWorkflowManager:
    def __init__(self, model_name: str = "llama3.2:1b"):
        """Initialize workflow manager"""
        self.current_state = AgentState()
        self.llm_client = OllamaClient(
            model_name=model_name
        )  # Pass model_name correctly
        self.s2_client = SemanticScholarClient()
        self.model_name = model_name
        self.search_limit = 10

    async def process_command_async(self, command: str) -> AgentState:
        """Process command and update state"""
        try:
            # Add command to state
            self.current_state.add_message("user", command)

            # Check if it's a search command
            if self._is_search_command(command.lower()):
                await self._handle_search(command)
            else:
                # Handle as normal conversation
                await self._handle_conversation(command)

            return self.current_state

        except Exception as e:
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            return self.current_state

    def _is_search_command(self, command: str) -> bool:
        """Check if the command is a search request"""
        search_triggers = [
            "search",
            "find",
            "look for",
            "papers about",
            "papers on",
            "research about",
            "research on",
        ]
        return any(trigger in command.lower() for trigger in search_triggers)

    async def _handle_conversation(self, command: str):
        """Handle normal conversation with LLM"""
        try:
            response = await self.llm_client.generate(
                prompt=command,
                system_prompt="You are a helpful AI assistant specializing in academic research. Help users with their questions and guide them to use the search function when they want to find papers.",
                max_tokens=300,
            )
            self.current_state.add_message("system", response.strip())
            self.current_state.status = AgentStatus.SUCCESS
        except Exception as e:
            self.current_state.add_message(
                "system", f"Sorry, I encountered an error: {str(e)}"
            )
            self.current_state.status = AgentStatus.ERROR

    async def _handle_search(self, command: str):
        """Handle search command with proper error handling"""
        try:
            # Clean and enhance query
            query = await self._enhance_query(command)

            # Perform search with retry logic
            max_retries = 3
            retry_delay = 2  # seconds

            for attempt in range(max_retries):
                try:
                    results = await self.s2_client.search_papers(
                        query=query, limit=self.search_limit  # Enforce limit
                    )
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    raise e

            # Update state with results
            self.current_state.update_search_results(results.papers, results.total)

            # Format response
            response = f"Found {results.total} papers. Here are the top {min(len(results.papers), self.search_limit)} most relevant ones:\n\n"
            for i, paper in enumerate(results.papers[: self.search_limit], 1):
                response += (
                    f"{i}. {paper.title} ({paper.year}) - {paper.citations} citations\n"
                )
            response += "\nYou can ask about any paper by its number or title, or ask me any other questions."

            self.current_state.add_message("system", response)

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                response = "I apologize, but I'm currently rate-limited by the academic search API. Please try again in a few moments."
            else:
                response = f"I apologize, but I encountered an error while searching: {error_msg}"

            self.current_state.add_message("system", response)
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = error_msg

    async def _enhance_query(self, query: str) -> str:
        """Enhance search query using LLM"""
        prompt = f"""Enhance this academic search query: "{query}"
        Remove conversational phrases and add relevant academic terms.
        Keep it focused and return only the enhanced search terms."""

        try:
            enhanced = await self.llm_client.generate(prompt=prompt, max_tokens=50)
            return enhanced.strip() or query
        except Exception:
            return query.replace("search", "").replace("find", "").strip()

    def reset_state(self):
        """Reset state to initial values"""
        self.current_state = AgentState()

    def get_state(self) -> AgentState:
        """Get current state"""
        return self.current_state
