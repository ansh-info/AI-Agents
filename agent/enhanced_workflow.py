import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from agent.workflow_graph import WorkflowGraph
from clients.ollama_client import OllamaClient
from clients.ollama_enhanced import EnhancedOllamaClient
from clients.semantic_scholar_client import (PaperMetadata, SearchFilters,
                                             SemanticScholarClient)
from state.agent_state import (AgentState, AgentStatus, ConversationMemory,
                               PaperContext, SearchContext)


class CommandParser:
    """Handles command parsing and intent detection"""

    @staticmethod
    def parse_command(command: str) -> Dict[str, Any]:
        """Parse command and identify intent"""
        print(f"[DEBUG] Parsing command: {command}")
        command_lower = command.lower()

        # Check for search intent
        if any(
            trigger in command_lower
            for trigger in [
                "search",
                "find",
                "look for",
                "papers about",
                "papers on",
                "research about",
                "can you search",
            ]
        ):
            cleaned_query = CommandParser._clean_search_query(command)
            print(f"[DEBUG] Detected search intent. Cleaned query: {cleaned_query}")
            return {
                "intent": "search",
                "query": cleaned_query,
            }

        # Check for paper question intent
        if any(
            trigger in command_lower
            for trigger in [
                "tell me about paper",
                "explain paper",
                "what does paper",
                "can you explain",
                "summarize paper",
            ]
        ):
            return {
                "intent": "paper_question",
                "paper_reference": CommandParser._extract_paper_reference(command),
            }

        # Check for comparison intent
        if any(
            trigger in command_lower
            for trigger in [
                "compare",
                "difference between",
                "how do papers",
                "similarities",
            ]
        ):
            return {
                "intent": "compare_papers",
                "paper_references": CommandParser._extract_multiple_papers(command),
            }

        # Default to conversation
        return {"intent": "conversation", "query": command}

    @staticmethod
    def _clean_search_query(query: str) -> str:
        """Clean search query for proper searching"""
        print(f"[DEBUG] Cleaning query: {query}")

        # List of phrases to remove
        clean_phrases = [
            "can you search for papers on",
            "can you search for papers about",
            "can you search for",
            "can you search",
            "search for papers on",
            "search for papers about",
            "search for papers",
            "search for",
            "papers on",
            "papers about",
            "can you",
            "papers",
            "search",
        ]

        cleaned = query.lower()
        # Remove phrases in order (longest first to prevent partial matches)
        for phrase in sorted(clean_phrases, key=len, reverse=True):
            cleaned = cleaned.replace(phrase, "")

        # Clean up extra whitespace
        cleaned = " ".join(cleaned.split())
        print(f"[DEBUG] Cleaned query result: {cleaned}")
        return cleaned

    @staticmethod
    def _extract_paper_reference(command: str) -> str:
        """Extract paper reference from command"""
        words = command.lower().split()
        for i, word in enumerate(words):
            if word == "paper" and i + 1 < len(words):
                if words[i + 1].isdigit():
                    return words[i + 1]
        return ""

    @staticmethod
    def _extract_multiple_papers(command: str) -> List[str]:
        """Extract multiple paper references"""
        references = []
        words = command.lower().split()
        for i, word in enumerate(words):
            if word == "paper" and i + 1 < len(words):
                if words[i + 1].isdigit():
                    references.append(words[i + 1])
        return references


class SearchAgent:
    """Handles paper search operations"""

    def __init__(self, s2_client: SemanticScholarClient):
        self.client = s2_client

    async def search_papers(
        self, query: str, filters: Optional[SearchFilters] = None
    ) -> Dict[str, Any]:
        """Perform paper search with enhanced error handling"""
        try:
            print(f"[DEBUG] SearchAgent: Starting search with query: {query}")
            if not query.strip():
                return {
                    "status": "error",
                    "results": None,
                    "error": "Empty search query",
                }

            results = await self.client.search_papers(
                query=query, filters=filters, limit=10
            )

            if not results or not results.papers:
                return {
                    "status": "success",
                    "results": SearchResults(total=0, offset=0, papers=[]),
                    "error": None,
                }

            return {"status": "success", "results": results, "error": None}

        except Exception as e:
            print(f"[DEBUG] SearchAgent: Error during search: {str(e)}")
            return {"status": "error", "results": None, "error": str(e)}

    async def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """Get detailed paper information"""
        try:
            details = await self.client.get_paper_details(paper_id)
            return {"status": "success", "paper": details, "error": None}
        except Exception as e:
            return {"status": "error", "paper": None, "error": str(e)}


class ConversationAgent:
    """Handles LLM-based conversations"""

    def __init__(self, ollama_client: OllamaClient):
        self.client = ollama_client

    async def generate_response(
        self, prompt: str, context: Optional[str] = None, max_tokens: int = 500
    ):
        """Generate response with enhanced context handling"""
        try:
            print(f"\n[DEBUG] Generating response")
            print(f"[DEBUG] Prompt: {prompt[:200]}...")
            if context:
                print(f"[DEBUG] Context: {context[:200]}...")

            # Build system context
            system_prompt = """You are a helpful academic research assistant. 
    When presenting search results:
    1. Always summarize papers that are provided in the context
    2. Focus on the specific papers mentioned, not general knowledge
    3. Highlight key findings from the actual papers
    4. Mention specific methodologies used in these papers
    5. Point out patterns across the provided papers"""

            if context:
                system_prompt += f"\n\nAdditional context: {context}"

            print(f"[DEBUG] Using system prompt: {system_prompt[:200]}...")

            response = await self.client.generate(
                prompt=prompt, system_prompt=system_prompt, max_tokens=max_tokens
            )

            print(f"[DEBUG] Generated response: {response[:200]}...")

            return {"status": "success", "response": response, "error": None}

        except Exception as e:
            print(f"[DEBUG] Error in generate_response: {str(e)}")
            return {"status": "error", "response": None, "error": str(e)}

    async def generate_paper_analysis(
        self, paper: PaperMetadata, question: str
    ) -> Dict[str, Any]:
        """Generate paper analysis"""
        try:
            context = (
                f"Title: {paper.title}\n"
                f"Authors: {', '.join(a.name for a in paper.authors)}\n"
                f"Year: {paper.year or 'N/A'}\n"
                f"Citations: {paper.citations or 0}\n"
                f"Abstract: {paper.abstract or 'Not available'}\n"
            )

            prompt = f"""Based on this academic paper:
{context}

Question: {question}

Please provide a comprehensive response that:
1. Addresses the specific question
2. References relevant parts of the paper
3. Provides context when needed
4. Mentions any uncertainties or limitations
5. Suggests related aspects to consider"""

            response = await self.generate_response(prompt)
            return response

        except Exception as e:
            return {"status": "error", "response": None, "error": str(e)}

    async def generate_paper_comparison(
        self, papers: List[PaperMetadata]
    ) -> Dict[str, Any]:
        """Generate paper comparison"""
        try:
            # Build context for all papers
            context = "Papers to compare:\n\n"
            for i, paper in enumerate(papers, 1):
                context += f"Paper {i}:\n"
                context += f"Title: {paper.title}\n"
                context += f"Authors: {', '.join(a.name for a in paper.authors)}\n"
                context += f"Year: {paper.year or 'N/A'}\n"
                context += f"Citations: {paper.citations or 0}\n"
                context += f"Abstract: {paper.abstract or 'Not available'}\n\n"

            prompt = f"""Please compare these papers, considering:
1. Main findings and contributions
2. Methodological approaches
3. Key differences and similarities
4. Relative impact and significance
5. How they complement or contrast with each other

{context}"""

            response = await self.generate_response(prompt, max_tokens=500)
            return response

        except Exception as e:
            return {"status": "error", "response": None, "error": str(e)}


class EnhancedWorkflowManager:
    """Main workflow manager with enhanced integration"""

    def __init__(self, model_name: str = "llama3.2:1b"):
        """Initialize workflow components"""
        self.ollama_client = OllamaClient(model_name=model_name)
        self.s2_client = SemanticScholarClient()

        # Initialize agents
        self.search_agent = SearchAgent(self.s2_client)
        self.conversation_agent = ConversationAgent(self.ollama_client)

        # Initialize workflow
        self.workflow_graph = WorkflowGraph(
            ollama_client=self.ollama_client, s2_client=self.s2_client
        )

        # Initialize state
        self.current_state = AgentState()

        # Command parser
        self.command_parser = CommandParser()

    async def process_command_async(self, command: str) -> AgentState:
        """Process command with enhanced monitoring"""
        try:
            # Add command to state
            print(f"\n[DEBUG] Processing command: {command}")
            self.current_state.add_message("user", command)

            # Parse command intent
            parsed_command = self.command_parser.parse_command(command)
            print(f"[DEBUG] Parsed command intent: {parsed_command['intent']}")

            # Get initial state debug info
            print("\n[DEBUG] Initial state:")
            initial_debug = await self.workflow_graph.debug_state(self.current_state)
            print(json.dumps(initial_debug, indent=2))

            # Process based on intent
            if parsed_command["intent"] == "search":
                await self._handle_search(parsed_command["query"])
            elif parsed_command["intent"] == "paper_question":
                await self._handle_paper_question(
                    parsed_command["paper_reference"], command
                )
            elif parsed_command["intent"] == "compare_papers":
                await self._handle_paper_comparison(parsed_command["paper_references"])
            else:
                await self._handle_conversation(command)

            # Process through workflow graph
            try:
                print("\n[DEBUG] Processing state through workflow graph...")
                self.current_state = await self.workflow_graph.process_state(
                    self.current_state
                )

                # Get final state debug info
                print("\n[DEBUG] Final state after processing:")
                final_debug = await self.workflow_graph.debug_state(self.current_state)
                print(json.dumps(final_debug, indent=2))

            except Exception as e:
                print(f"[DEBUG] Workflow error: {str(e)}")
                self.current_state.status = AgentStatus.ERROR
                self.current_state.error_message = f"Workflow error: {str(e)}"

            return self.current_state

        except Exception as e:
            print(f"[DEBUG] Command processing error: {str(e)}")
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            self.current_state.add_message(
                "system", f"I apologize, but I encountered an error: {str(e)}"
            )
            return self.current_state

    async def check_workflow_health(self) -> Dict[str, Any]:
        """Check the health of the workflow components"""
        health_status = {
            "workflow_graph": True,
            "ollama_client": False,
            "semantic_scholar": False,
            "command_parser": True,
            "errors": [],
        }

        try:
            # Check Ollama
            ollama_health = await self.ollama_client.check_model_availability()
            health_status["ollama_client"] = ollama_health

            # Check Semantic Scholar
            s2_health = await self.s2_client.check_api_status()
            health_status["semantic_scholar"] = s2_health

            if not ollama_health:
                health_status["errors"].append("Ollama client is not responding")
            if not s2_health:
                health_status["errors"].append("Semantic Scholar API is not responding")

        except Exception as e:
            health_status["errors"].append(f"Health check error: {str(e)}")

        print("[DEBUG] Workflow health status:")
        print(json.dumps(health_status, indent=2))

        return health_status

    async def _handle_search(self, query: str):
        """Handle search with proper state management"""
        try:
            # Clean query
            clean_query = query.replace("?", "").strip()
            if not clean_query:
                raise ValueError("Search query cannot be empty")

            print(f"[DEBUG] Processing search for query: '{clean_query}'")

            # Initialize state for search
            self.current_state.update_state(
                status=AgentStatus.PROCESSING,
                current_step="search_started",
                next_steps=["process_search"],
                last_update=datetime.now(),
            )

            # Perform search
            search_result = await self.search_agent.search_papers(clean_query)

            if search_result["status"] == "error":
                raise Exception(f"Search failed: {search_result['error']}")

            results = search_result["results"]

            # Update search context
            self.current_state.search_context.query = clean_query

            # Format results with proper structure
            formatted_papers = []
            for i, paper in enumerate(results.papers, 1):
                formatted_paper = {
                    "number": i,
                    "title": paper.title,
                    "authors": ", ".join(a.name for a in paper.authors),
                    "year": paper.year,
                    "citations": paper.citations,
                    "abstract": (
                        paper.abstract[:300] + "..."
                        if paper.abstract
                        else "No abstract available"
                    ),
                    "url": paper.url,
                }
                formatted_papers.append(formatted_paper)

                # Add to search context
                paper_data = {
                    "paperId": paper.paperId,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "year": paper.year,
                    "authors": [
                        {"name": a.name, "authorId": a.authorId} for a in paper.authors
                    ],
                    "citations": paper.citations,
                    "url": paper.url,
                }
                self.current_state.search_context.add_paper(paper_data)

            # Build message content with markdown formatting
            message_parts = [
                f"# Search Results\nFound {results.total} papers related to '{clean_query}'.\n"
            ]

            # Add each paper with proper formatting
            for paper in formatted_papers:
                message_parts.extend(
                    [
                        f"## {paper['number']}. {paper['title']}",
                        f"**Authors:** {paper['authors']}",
                        f"**Year:** {paper['year'] or 'N/A'} | **Citations:** {paper['citations'] or 0}",
                        "",
                        f"**Abstract:** {paper['abstract']}",
                        f"[View Paper]({paper['url']})\n",
                        "---\n",
                    ]
                )

            # Generate and add summary
            summary_context = [
                {
                    "title": paper["title"],
                    "year": paper["year"],
                    "citations": paper["citations"],
                    "abstract_preview": (
                        paper["abstract"][:200] if paper["abstract"] else "No abstract"
                    ),
                }
                for paper in formatted_papers
            ]

            try:
                summary_response = await self.conversation_agent.generate_response(
                    prompt=f"""Based on these papers about "{clean_query}", please provide a brief summary of:
                    1. Main research areas covered
                    2. Key methodologies mentioned
                    3. Notable findings
                    4. Visible trends
                    
                    Papers: {json.dumps(summary_context)}""",
                    max_tokens=300,
                )

                if summary_response["status"] == "success":
                    summary = summary_response["response"]
                    message_parts.extend(["\n## Summary", summary])
                else:
                    message_parts.append("\n*Unable to generate summary at this time.*")
            except Exception as e:
                print(f"[DEBUG] Summary generation error: {str(e)}")
                message_parts.append("\n*Unable to generate summary at this time.*")

            # Update state
            self.current_state.update_state(
                status=AgentStatus.SUCCESS,
                current_step="search_completed",
                next_steps=["update_memory"],
                last_update=datetime.now(),
            )

            # Add the formatted message
            self.current_state.add_message("system", "\n".join(message_parts))

            # Ensure state history is updated
            if not hasattr(self.current_state, "state_history"):
                self.current_state.state_history = []
            self.current_state.state_history.append(
                {
                    "timestamp": datetime.now(),
                    "step": "search_completed",
                    "query": clean_query,
                    "results_count": len(results.papers),
                }
            )

            return {"state": self.current_state, "next": "update_memory"}

        except Exception as e:
            print(f"[DEBUG] Search handling error: {str(e)}")
            self.current_state.update_state(
                status=AgentStatus.ERROR,
                error_message=str(e),
                current_step="error",
                next_steps=["update_memory"],
                last_update=datetime.now(),
            )

            # Ensure state history is updated even in error case
            if not hasattr(self.current_state, "state_history"):
                self.current_state.state_history = []
            self.current_state.state_history.append(
                {"timestamp": datetime.now(), "step": "error", "error": str(e)}
            )

            self.current_state.add_message(
                "system", f"I encountered an error while searching: {str(e)}"
            )
            return {"state": self.current_state, "next": "update_memory"}

    async def _handle_paper_question(self, paper_reference: str, question: str):
        """Handle paper-specific questions"""
        try:
            # Get paper from context
            paper = self._get_paper_by_reference(paper_reference)
            if not paper:
                self.current_state.add_message(
                    "system",
                    "I'm not sure which paper you're referring to. Could you specify the paper number?",
                )
                return

            # Generate analysis
            analysis = await self.conversation_agent.generate_paper_analysis(
                paper, question
            )

            if analysis["status"] == "error":
                raise Exception(f"Analysis failed: {analysis['error']}")

            self.current_state.memory.focused_paper = paper
            self.current_state.add_message("system", analysis["response"])
            self.current_state.status = AgentStatus.SUCCESS

        except Exception as e:
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            self.current_state.add_message(
                "system",
                f"I apologize, but I encountered an error while analyzing the paper: {str(e)}",
            )

    async def _handle_paper_comparison(self, paper_references: List[str]):
        """Handle paper comparison requests"""
        try:
            papers = [self._get_paper_by_reference(ref) for ref in paper_references]
            papers = [p for p in papers if p]  # Filter None values

            if len(papers) < 2:
                self.current_state.add_message(
                    "system",
                    "I need at least two valid paper references to make a comparison.",
                )
                return

            comparison = await self.conversation_agent.generate_paper_comparison(papers)

            if comparison["status"] == "error":
                raise Exception(f"Comparison failed: {comparison['error']}")

            self.current_state.add_message("system", comparison["response"])
            self.current_state.status = AgentStatus.SUCCESS

        except Exception as e:
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            self.current_state.add_message(
                "system",
                f"I apologize, but I encountered an error while comparing the papers: {str(e)}",
            )

    async def _handle_conversation(self, command: str):
        """Handle general conversation"""
        try:
            # Build conversation context
            context = self._build_conversation_context()

            response = await self.conversation_agent.generate_response(
                prompt=command, context=context
            )

            if response["status"] == "error":
                raise Exception(f"Response generation failed: {response['error']}")

            self.current_state.add_message("system", response["response"])
            self.current_state.status = AgentStatus.SUCCESS

        except Exception as e:
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            self.current_state.add_message(
                "system", f"I apologize, but I encountered an error: {str(e)}"
            )

    def _get_paper_by_reference(self, reference: str) -> Optional[PaperMetadata]:
        """Get paper by reference number"""
        try:
            if reference.isdigit():
                index = int(reference) - 1
                if 0 <= index < len(self.current_state.search_context.results):
                    return self.current_state.search_context.results[index]
            return None
        except Exception:
            return None

    def _build_conversation_context(self) -> str:
        """Build context for conversation"""
        context_parts = []

        # Add recent conversation history
        recent_messages = self.current_state.memory.messages[-5:]
        if recent_messages:
            context_parts.append("Recent conversation:")
            for msg in recent_messages:
                context_parts.append(f"{msg['role']}: {msg['content']}")

        # Add search context if available
        if self.current_state.search_context.results:
            context_parts.append("\nAvailable papers:")
            for i, paper in enumerate(self.current_state.search_context.results, 1):
                context_parts.append(
                    f"{i}. {paper.title} ({paper.year or 'N/A'}) - {paper.citations or 0} citations"
                )

        # Add focused paper if available
        if self.current_state.memory.focused_paper:
            paper = self.current_state.memory.focused_paper
            context_parts.append(f"\nCurrently discussing paper: {paper.title}")

        return "\n".join(context_parts)

    def get_state(self) -> AgentState:
        """Get current state"""
        return self.current_state

    def reset_state(self):
        """Reset state to initial values"""
        self.current_state = AgentState()

    async def check_clients_health(self) -> Dict[str, bool]:
        """Check if all clients are healthy and responding"""
        try:
            # Check Ollama
            ollama_health = await self.ollama_client.check_model_availability()

            # Check Semantic Scholar
            s2_health = await self.s2_client.check_api_status()

            return {
                "ollama_status": ollama_health,
                "semantic_scholar_status": s2_health,
                "all_healthy": ollama_health and s2_health,
            }
        except Exception:
            return {
                "ollama_status": False,
                "semantic_scholar_status": False,
                "all_healthy": False,
            }

    async def reload_clients(self) -> bool:
        """Attempt to reload clients if they're not responding"""
        try:
            # Reinitialize clients
            self.ollama_client = OllamaClient()
            self.s2_client = SemanticScholarClient()

            # Reinitialize agents
            self.search_agent = SearchAgent(self.s2_client)
            self.conversation_agent = ConversationAgent(self.ollama_client)

            # Check health
            health = await self.check_clients_health()
            return health["all_healthy"]
        except Exception:
            return False
