import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from agent.workflow_graph import WorkflowGraph
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
        self.workflow_graph = WorkflowGraph()

    async def process_command_async(self, command: str) -> AgentState:
        """Process command using LangGraph workflow"""
        try:
            # Add command to state
            self.current_state.add_message("user", command)

            # Process based on command type
            if self._is_search_command(command.lower()):
                await self._handle_search(command)
            elif self._is_paper_question(command.lower()):
                await self._handle_paper_question(
                    self._get_paper_reference(command), command
                )
            else:
                await self._handle_conversation(command)

            # Process through the workflow graph
            try:
                processed_state = self.workflow_graph.process_state(self.current_state)
                if processed_state:
                    self.current_state = processed_state
            except Exception as e:
                self.current_state.status = AgentStatus.ERROR
                self.current_state.error_message = str(e)
            return self.current_state

        except Exception as e:
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            self.current_state.add_message(
                "system", f"I apologize, but I encountered an error: {str(e)}"
            )
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

    def _is_paper_question(self, command: str) -> bool:
        """Check if the command is asking about a paper"""
        if not self.current_state.search_context.results:
            return False
        paper_indicators = [
            "paper",
            "study",
            "research",
            "article",
            "what is",
            "tell me about",
        ]
        return any(indicator in command.lower() for indicator in paper_indicators)

    def _get_paper_reference(self, command: str) -> Optional[PaperContext]:
        """Extract paper reference from command"""
        if not self.current_state.search_context.results:
            return None

        command_lower = command.lower()

        # Check for numeric references
        words = command_lower.split()
        for i, word in enumerate(words):
            if word in ["paper", "study", "article"] and i > 0:
                try:
                    num = int(words[i - 1])
                    if 1 <= num <= len(self.current_state.search_context.results):
                        return self.current_state.search_context.results[num - 1]
                except (ValueError, IndexError):
                    continue

        # Check for ordinal references
        ordinals = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}
        for ordinal, idx in ordinals.items():
            if ordinal in command_lower and idx < len(
                self.current_state.search_context.results
            ):
                return self.current_state.search_context.results[idx]

        # Check for title references
        for paper in self.current_state.search_context.results:
            if paper.title.lower() in command_lower:
                return paper

        return None

    async def _handle_paper_question(
        self, paper: Optional[PaperContext], question: str
    ):
        """Handle questions about a specific paper"""
        if not paper:
            self.current_state.add_message(
                "system",
                "I'm not sure which paper you're referring to. Could you specify the paper number or title?",
            )
            return

        paper_context = (
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(a.get('name', '') for a in paper.authors)}\n"
            f"Year: {paper.year or 'Not specified'}\n"
            f"Citations: {paper.citations or 0}\n"
        )
        if paper.abstract:
            paper_context += f"Abstract: {paper.abstract}\n"

        prompt = f"""Based on this paper:
{paper_context}

Question: {question}
Please provide a comprehensive answer based on the available information."""

        try:
            response = await self.llm_client.generate(prompt=prompt, max_tokens=300)
            self.current_state.memory.focused_paper = paper
            self.current_state.add_message("system", response.strip())
            self.current_state.status = AgentStatus.SUCCESS
        except Exception as e:
            self.current_state.add_message(
                "system",
                f"I apologize, but I encountered an error while processing your question: {str(e)}",
            )
            self.current_state.status = AgentStatus.ERROR

    async def _handle_search(self, command: str):
        """Handle search command with proper error handling"""
        try:
            # Clean query
            query = self._clean_search_query(command)

            results = await self.s2_client.search_papers(
                query=query, limit=self.search_limit
            )

            # Update state
            self.current_state.status = AgentStatus.SUCCESS
            self.current_state.search_context.query = query
            self.current_state.search_context.total_results = results.total
            self.current_state.search_context.results = []

            # Add papers to context
            for paper in results.papers:
                # Convert paper data to correct format
                paper_data = {
                    "paperId": paper.paperId,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "year": paper.year,
                    "authors": [
                        {"name": author.name, "authorId": author.authorId}
                        for author in paper.authors
                    ],
                    "citations": paper.citations,
                    "url": paper.url,
                }
                self.current_state.search_context.add_paper(paper_data)

            # Format response with a table
            response = f"Found {results.total} papers. Here are the top {len(self.current_state.search_context.results)} most relevant ones:\n\n"
            response += "| # | Title | Year | Citations |\n"
            response += "|---|--------|------|------------|\n"

            for i, paper in enumerate(self.current_state.search_context.results, 1):
                title = paper.title.replace("|", "-")  # Escape pipe characters
                response += f"| {i} | {title} | {paper.year or 'N/A'} | {paper.citations or 0} |\n"

            response += "\nYou can ask me about any of these papers by number or title. For example:\n"
            response += "- Tell me more about paper 1\n"
            response += "- What is paper 2 about?\n"
            response += "- Can you explain the first paper's findings?"

            self.current_state.add_message("system", response)

        except Exception as e:
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            self.current_state.add_message(
                "system",
                f"I apologize, but I encountered an error while searching: {str(e)}",
            )

    async def _handle_conversation(self, command: str):
        """Handle normal conversation with LLM"""
        context = []

        # Add recent conversation history
        recent_messages = self.current_state.memory.messages[-5:]  # Last 5 messages
        if recent_messages:
            context.append("Recent conversation:")
            for msg in recent_messages:
                context.append(f"{msg['role']}: {msg['content']}")

        # Add search results context if available
        if self.current_state.search_context.results:
            context.append("\nCurrent search results:")
            for i, paper in enumerate(self.current_state.search_context.results, 1):
                context.append(
                    f"{i}. {paper.title} ({paper.year or 'N/A'}) - {paper.citations or 0} citations"
                )

        # Add paper context if available
        if self.current_state.memory.focused_paper:
            paper = self.current_state.memory.focused_paper
            context.append(f"\nCurrently discussing paper: {paper.title}")

        prompt = f"""Context:
{chr(10).join(context)}

Current query: {command}
The user is asking about the search results or papers. Please provide a helpful response based on the available context. 
If they ask about specific papers, refer to them by their details from the search results."""

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                system_prompt="You are a helpful AI assistant specializing in academic research.",
                max_tokens=300,
            )
            self.current_state.add_message("system", response.strip())
            self.current_state.status = AgentStatus.SUCCESS
        except Exception as e:
            self.current_state.add_message(
                "system", f"Sorry, I encountered an error: {str(e)}"
            )
            self.current_state.status = AgentStatus.ERROR

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

    def get_state(self) -> AgentState:
        """Get current state"""
        return self.current_state

    def reset_state(self):
        """Reset state to initial values"""
        self.current_state = AgentState()
