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
                               PaperContext, SearchContext)


class EnhancedWorkflowManager:
    def __init__(self, model_name: str = "llama3.2:1b"):
        """Initialize workflow manager"""
        self.current_state = AgentState()
        self.llm_client = OllamaClient(model_name=model_name)
        self.s2_client = SemanticScholarClient()
        self.model_name = model_name
        self.search_limit = 10

    async def process_command_async(self, command: str) -> AgentState:
        """Process command and update state"""
        try:
            # Add command to state
            self.current_state.add_message("user", command)

            # First check if it's a question about a specific paper
            if self.current_state.search_context.results:
                paper_ref = self._get_paper_reference(command)
                if paper_ref is not None:
                    await self._handle_paper_question(paper_ref, command)
                    return self.current_state

            # If not a paper question, check if it's a search command
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

    def _get_paper_reference(self, command: str) -> Optional[PaperContext]:
        """Extract paper reference from command"""
        command_lower = command.lower()

        # Check for numeric references
        if "paper" in command_lower or "the" in command_lower:
            words = command_lower.split()
            for i, word in enumerate(words):
                if word in ["paper", "the"] and i > 0:
                    try:
                        num = int(words[i - 1])
                        if 1 <= num <= len(self.current_state.search_context.results):
                            return self.current_state.search_context.results[num - 1]
                    except (ValueError, IndexError):
                        pass

        # Check for ordinal references
        ordinals = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
        for ordinal, num in ordinals.items():
            if ordinal in command_lower and num <= len(
                self.current_state.search_context.results
            ):
                return self.current_state.search_context.results[num - 1]

        return None

    async def _handle_paper_question(self, paper: PaperContext, question: str):
        """Handle questions about a specific paper"""
        # Create context about the paper
        paper_context = (
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(a['name'] for a in paper.authors)}\n"
            f"Year: {paper.year or 'Not specified'}\n"
            f"Citations: {paper.citations or 0}\n"
        )
        if paper.abstract:
            paper_context += f"Abstract: {paper.abstract}\n"

        prompt = f"""Based on this paper:
{paper_context}

Please answer this question: {question}

If the question is asking for details not provided in the context, focus on what information is available and mention if specific details are not provided."""

        response = await self.llm_client.generate(prompt=prompt, max_tokens=300)

        self.current_state.memory.focused_paper = paper
        self.current_state.add_message("system", response.strip())
        self.current_state.status = AgentStatus.SUCCESS

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
            # Clean query
            query = self._clean_search_query(command)

            # Perform search
            try:
                results = await self.s2_client.search_papers(
                    query=query, limit=self.search_limit
                )

                # Convert Semantic Scholar papers to PaperContext
                paper_contexts = []
                for paper in results.papers:
                    paper_context = PaperContext(
                        paper_id=paper.paperId,
                        title=paper.title,
                        authors=[{"name": author.name} for author in paper.authors],
                        year=paper.year,
                        citations=paper.citations,
                        abstract=paper.abstract,
                        url=paper.url,
                    )
                    paper_contexts.append(paper_context)

                # Update state
                self.current_state.status = AgentStatus.SUCCESS
                self.current_state.search_context.results = paper_contexts
                self.current_state.search_context.total_results = results.total

                # Format response
                response = f"Found {results.total} papers. Here are the top {len(paper_contexts)} most relevant ones:\n\n"
                for i, paper in enumerate(paper_contexts, 1):
                    response += f"{i}. {paper.title} ({paper.year or 'N/A'}) - {paper.citations or 0} citations\n"
                response += (
                    "\nYou can ask me about any of these papers by number or title."
                )

                self.current_state.add_message("system", response)

            except Exception as e:
                raise Exception(f"Search failed: {str(e)}")

        except Exception as e:
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            self.current_state.add_message(
                "system",
                f"I apologize, but I encountered an error while searching: {str(e)}",
            )

    def _clean_search_query(self, query: str) -> str:
        """Clean the search query"""
        clean_phrases = [
            "can you find me",
            "find me",
            "search for",
            "papers on",
            "papers about",
            "paper on",
            "paper about",
        ]
        cleaned_query = query.lower()
        for phrase in clean_phrases:
            cleaned_query = cleaned_query.replace(phrase, "")
        return cleaned_query.strip()

    def reset_state(self):
        """Reset state to initial values"""
        self.current_state = AgentState()

    def get_state(self) -> AgentState:
        """Get current state"""
        return self.current_state
