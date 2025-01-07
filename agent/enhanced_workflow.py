import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from clients.ollama_client import OllamaClient
from clients.semantic_scholar_client import (SearchResults,
                                             SemanticScholarClient)
from state.agent_state import (AgentState, AgentStatus, ConversationMemory,
                               SearchContext)


class EnhancedWorkflowManager:
    def __init__(self, model_name: str = "llama3.2:1b"):
        """Initialize workflow manager"""
        self.current_state = AgentState()
        self.llm_client = OllamaClient()
        self.s2_client = SemanticScholarClient()
        self.model_name = model_name
        self.search_limit = 10  # Default number of results per page

    async def process_command_async(self, command: str) -> AgentState:
        """Process command and update state asynchronously"""
        try:
            print(f"Debug: Processing command: {command}")

            # Add command to state without resetting
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
        command_lower = command.lower().strip()

        if command_lower == "help":
            state.memory.current_context = "HELP"
            state.current_step = "help_displayed"
        elif command_lower.startswith("search"):
            state.memory.current_context = "SEARCH"
            state.current_step = "search_initiated"
            # Reset search context for new search
            state.search_context = SearchContext()
        elif command_lower.startswith("next"):
            state.memory.current_context = "PAGINATION"
            state.current_step = "next_page"
        elif command_lower.startswith("prev"):
            state.memory.current_context = "PAGINATION"
            state.current_step = "prev_page"
        elif command_lower == "clear":
            # Store the command before reset
            command_msg = {"role": "user", "content": command}
            self.reset_state()
            # Restore the command after reset
            self.current_state.memory.messages.append(command_msg)
            self.current_state.memory.current_context = "CLEAR"
            self.current_state.current_step = "cleared"
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
2. next: Show next page of search results
3. prev: Show previous page of search results
4. clear: Clear current search and state
5. help: Show this help message

Example usage:
- search papers about machine learning
- search recent LLM papers
- next
- prev
- clear
- help"""
            state.add_message("system", help_text)
            state.current_step = "help_displayed"

        elif command_type == "SEARCH":
            await self._handle_search(state, command)

        elif command_type == "PAGINATION":
            await self._handle_pagination(state, command)

        elif command_type == "CLEAR":
            state.add_message("system", "Search state cleared.")
            state.current_step = "cleared"

        else:
            state.add_message(
                "system",
                f"Unknown command: {command}. Type 'help' for available commands.",
            )
            state.current_step = "command_processed"

        state.status = AgentStatus.SUCCESS

    async def _parse_search_command(self, command: str) -> dict:
        """Parse search command with filters
        Example inputs:
        - search transformers year:2023
        - search llm citations>1000
        - search neural networks year:2020-2023 citations>500
        """
        # Remove 'search' from the start
        query_text = command.replace("search", "", 1).strip()

        filters = {
            "query": "",
            "year": None,
            "year_range": None,
            "min_citations": None,
            "sort_by": None,
            "sort_order": "desc",
        }

        parts = query_text.split()
        main_query = []

        for part in parts:
            if ":" in part:
                key, value = part.split(":", 1)
                if key == "year":
                    if "-" in value:
                        start, end = value.split("-")
                        filters["year_range"] = (int(start), int(end))
                    else:
                        filters["year"] = int(value)
            elif ">" in part and "citations" in part:
                filters["min_citations"] = int(part.split(">")[1])
            elif part.startswith("sort:"):
                filters["sort_by"] = part.split(":")[1]
            else:
                main_query.append(part)

        filters["query"] = " ".join(main_query)
        return filters

    async def _handle_search(self, state: AgentState, command: str) -> None:
        """Handle search command and fetch results"""
        if command == "search":
            state.add_message(
                "system", "Please provide a search query. Example: 'search LLM papers'"
            )
            state.current_step = "invalid_search"
            return

        # Extract query by removing 'search' prefix
        query = command.replace("search", "", 1).strip()

        try:
            # Perform search
            results = await self.s2_client.search_papers(
                query=query, limit=self.search_limit, offset=0
            )

            # Update state with results
            state.search_context.query = query
            state.search_context.current_page = 1
            state.search_context.total_results = results.total
            state.search_context.results = results.papers

            # Format results message
            result_message = await self._format_search_results(results)
            state.add_message("system", result_message)
            state.current_step = "search_completed"

        except Exception as e:
            state.add_message("system", f"Error performing search: {str(e)}")
            state.current_step = "search_error"
            state.status = AgentStatus.ERROR

    async def _handle_pagination(self, state: AgentState, command: str) -> None:
        """Handle pagination commands (next/prev)"""
        if not state.search_context.query:
            state.add_message(
                "system", "No active search. Please perform a search first."
            )
            return

        current_page = state.search_context.current_page or 1
        if command.startswith("next"):
            offset = current_page * self.search_limit
            if offset >= state.search_context.total_results:
                state.add_message("system", "No more results available.")
                return
            new_page = current_page + 1
        else:  # prev
            if current_page <= 1:
                state.add_message("system", "Already on the first page.")
                return
            offset = (current_page - 2) * self.search_limit
            new_page = current_page - 1

        try:
            # Get stored filters
            filters = getattr(state.search_context, "current_filters", {})

            # Prepare API parameters
            params = {
                "query": state.search_context.query,
                "limit": self.search_limit,
                "offset": offset,
            }

            # Apply stored filters
            if filters.get("year"):
                params["year"] = filters["year"]
            elif filters.get("year_range"):
                start, end = filters["year_range"]
                params["query"] += f" year>={start} year<={end}"

            results = await self.s2_client.search_papers(**params)

            # Apply post-processing filters
            if filters.get("min_citations"):
                results.papers = [
                    paper
                    for paper in results.papers
                    if paper.citationCount
                    and paper.citationCount >= filters["min_citations"]
                ]

            # Apply sorting
            if filters.get("sort_by"):
                reverse = filters.get("sort_order", "desc") == "desc"
                if filters["sort_by"] == "citations":
                    results.papers.sort(
                        key=lambda x: x.citationCount or 0, reverse=reverse
                    )
                elif filters["sort_by"] == "year":
                    results.papers.sort(key=lambda x: x.year or 0, reverse=reverse)

            state.search_context.current_page = new_page
            state.search_context.results = results.papers

            result_message = await self._format_search_results(results)
            state.add_message("system", result_message)
            state.current_step = "pagination_completed"

        except Exception as e:
            state.add_message("system", f"Error fetching results: {str(e)}")
            state.current_step = "pagination_error"
            state.status = AgentStatus.ERROR

    async def _format_search_results(self, results: SearchResults) -> str:
        """Format search results for display"""
        start_idx = (results.offset or 0) + 1
        result_message = f"Found {results.total} papers. Showing results {start_idx}-{start_idx + len(results.papers) - 1}:\n\n"

        for i, paper in enumerate(results.papers, start_idx):
            authors = ", ".join(a.get("name", "") for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."

            result_message += f"{i}. {paper.title} ({paper.year})\n"
            result_message += f"   Authors: {authors}\n"
            if paper.abstract:
                result_message += f"   Abstract: {paper.abstract[:200]}...\n"
            result_message += f"   Citations: {paper.citationCount}\n\n"

        return result_message

    def process_command_external(self, command: str) -> AgentState:
        """Synchronous interface for command processing"""
        return asyncio.run(self.process_command_async(command))

    def get_state(self) -> AgentState:
        """Get current state"""
        return self.current_state

    def reset_state(self):
        """Reset state to initial values"""
        self.current_state = AgentState()
