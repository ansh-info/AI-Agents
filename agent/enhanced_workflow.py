import asyncio
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
        """Clean search query"""
        print(f"[DEBUG] Cleaning query: {query}")
        clean_phrases = [
            "search for",
            "find me",
            "look for",
            "papers about",
            "papers on",
            "research on",
            "can you find",
            "can you search for",
            "can you search",
            "search",
        ]
        cleaned = query.lower()
        for phrase in clean_phrases:
            cleaned = cleaned.replace(phrase, "")
        cleaned = cleaned.strip()
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
        """Perform paper search with error handling"""
        try:
            print(f"[DEBUG] SearchAgent: Starting search with query: {query}")
            results = await self.client.search_papers(
                query=query, filters=filters, limit=10
            )

            print(f"[DEBUG] SearchAgent: Got {results.total} total results")
            print("[DEBUG] SearchAgent: First paper details:")
            if results.papers:
                first_paper = results.papers[0]
                print(f"  - paperId: {first_paper.paperId}")
                print(f"  - title: {first_paper.title}")
                print(f"  - authors: {[a.name for a in first_paper.authors]}")

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

            # Build system context
            system_prompt = """You are a helpful academic research assistant. 
When presenting search results:
1. Always summarize the key findings
2. Highlight the relevance to the search query
3. Mention important details like methodologies and conclusions
4. Point out any significant patterns or trends
5. Suggest related areas of interest"""

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
        """Process command with enhanced error handling and logging"""
        try:
            print("\n[DEBUG] Starting command processing")
            print(f"[DEBUG] Command received: {command}")

            # Parse command intent
            parsed_command = self.command_parser.parse_command(command)
            print(f"[DEBUG] Parsed command intent: {parsed_command['intent']}")

            # Add command to state
            self.current_state.add_message("user", command)

            if parsed_command["intent"] == "search":
                print("[DEBUG] Handling search intent")
                await self._handle_search(parsed_command["query"])

                # After search, check state
                print(
                    f"[DEBUG] After search - State status: {self.current_state.status}"
                )
                print(
                    f"[DEBUG] Results count: {len(self.current_state.search_context.results)}"
                )
                print("[DEBUG] Checking memory messages:")
                for msg in self.current_state.memory.messages[-3:]:
                    print(f"  - {msg['role']}: {msg['content'][:100]}...")

                if self.current_state.search_context.results:
                    # Generate a summary response that includes the papers
                    papers_summary = f"I found {len(self.current_state.search_context.results)} papers related to your query. Here are the results:\n\n"
                    for i, paper in enumerate(
                        self.current_state.search_context.results, 1
                    ):
                        papers_summary += f"{i}. {paper.title}\n"
                        papers_summary += f"   Authors: {', '.join(a.get('name', '') for a in paper.authors)}\n"
                        if paper.year:
                            papers_summary += f"   Year: {paper.year}\n"
                        if paper.abstract:
                            papers_summary += (
                                f"   Abstract: {paper.abstract[:200]}...\n"
                            )
                        papers_summary += "\n"

                    print("[DEBUG] Adding papers summary to state")
                    self.current_state.add_message("system", papers_summary)

            elif parsed_command["intent"] == "paper_question":
                print("[DEBUG] Handling paper question intent")
                await self._handle_paper_question(
                    parsed_command["paper_reference"], command
                )

            elif parsed_command["intent"] == "compare_papers":
                print("[DEBUG] Handling paper comparison intent")
                await self._handle_paper_comparison(parsed_command["paper_references"])

            else:
                print("[DEBUG] Handling conversation intent")
                await self._handle_conversation(command)

            return self.current_state

        except Exception as e:
            print(f"[DEBUG] Error in process_command_async: {str(e)}")
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            self.current_state.add_message(
                "system",
                f"I apologize, but I encountered an error: {str(e)}",
            )
            return self.current_state

    async def _handle_search(self, query: str):
        """Handle search with error recovery"""
        try:
            print("\n[DEBUG] ===== Search Process Start =====")

            # Perform search
            search_result = await self.search_agent.search_papers(query)
            print(f"[DEBUG] Search result status: {search_result['status']}")

            if search_result["status"] == "error":
                raise Exception(f"Search failed: {search_result['error']}")

            results = search_result["results"]
            print(f"[DEBUG] Total results: {results.total}")

            # Update state
            self.current_state.status = AgentStatus.SUCCESS
            self.current_state.search_context.query = query
            self.current_state.search_context.total_results = results.total

            # Add papers to context
            papers_added = 0
            papers_text = "Here are the papers I found:\n\n"

            for i, paper in enumerate(results.papers, 1):
                try:
                    paper_data = {
                        "paperId": paper.paperId,
                        "title": paper.title,
                        "abstract": paper.abstract,
                        "year": paper.year,
                        "authors": [
                            {"name": a.name, "authorId": a.authorId}
                            for a in paper.authors
                        ],
                        "citations": paper.citations,
                        "url": paper.url,
                    }

                    print(f"[DEBUG] Processing paper {i}: {paper_data['title']}")

                    self.current_state.search_context.add_paper(paper_data)
                    papers_added += 1

                    # Build text representation for LLM
                    papers_text += f"{i}. {paper_data['title']}\n"
                    papers_text += f"   Authors: {', '.join(a['name'] for a in paper_data['authors'])}\n"
                    papers_text += f"   Year: {paper_data['year']}\n"
                    if paper_data["abstract"]:
                        papers_text += (
                            f"   Abstract: {paper_data['abstract'][:200]}...\n"
                        )
                    papers_text += "\n"

                except Exception as e:
                    print(f"[DEBUG] Error processing paper {i}: {str(e)}")
                    continue

            print(f"[DEBUG] Added {papers_added} papers to context")

            if papers_added > 0:
                # Generate summary using LLM
                prompt = f"""I found {papers_added} papers related to '{query}'. 

    {papers_text}

    Please provide a helpful summary of these papers, highlighting key findings and relevance to the search query."""

                print("[DEBUG] Generating response with papers")
                response = await self.conversation_agent.generate_response(
                    prompt=prompt
                )

                if response["status"] == "error":
                    raise Exception(f"Response generation failed: {response['error']}")

                # Add both the summary and the papers list to the response
                summary = response["response"]
                full_response = f"{summary}\n\n{papers_text}"

                print("[DEBUG] Adding response to state")
                self.current_state.add_message("system", full_response)
            else:
                error_msg = "No valid papers found matching your search criteria."
                self.current_state.add_message("system", error_msg)

            print("[DEBUG] ===== Search Process Complete =====")

        except Exception as e:
            print(f"[DEBUG] Error in search handler: {str(e)}")
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            self.current_state.add_message(
                "system",
                f"I apologize, but I encountered an error while searching: {str(e)}",
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
