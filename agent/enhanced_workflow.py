import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent.main_agent import MainAgent
from agent.workflow_graph import WorkflowGraph
from agent.workflow_manager import ResearchWorkflowManager
from clients.ollama_client import OllamaClient
from clients.semantic_scholar_client import (
    PaperMetadata,
    SearchFilters,
    SemanticScholarClient,
)
from state.agent_state import AgentState, AgentStatus
from tools.ollama_tool import OllamaTool
from tools.paper_analyzer_tool import PaperAnalyzerTool
from tools.semantic_scholar_tool import SemanticScholarTool


class CommandParser:
    """Handles command parsing and intent detection"""

    @classmethod
    def parse_command(cls, command: str) -> Dict[str, Any]:
        """Parse command and determine intent"""
        command = command.lower().strip()

        # Determine intent
        if any(
            word in command for word in ["find", "search", "look for", "papers about"]
        ):
            return {"intent": "search", "query": command}
        elif "analyze" in command or "explain" in command:
            return {"intent": "analyze", "query": command}
        else:
            return {"intent": "conversation", "query": command}

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


class ConversationAgent:
    """Handles LLM-based conversations"""

    def __init__(self, ollama_client: OllamaClient):
        """Initialize conversation agent"""
        try:
            print("[DEBUG] Initializing ConversationAgent")
            if not ollama_client:
                raise ValueError("OllamaClient cannot be None")
            self.client = ollama_client
            print("[DEBUG] ConversationAgent initialized successfully")
        except Exception as e:
            print(f"[DEBUG] Error initializing ConversationAgent: {str(e)}")
            raise

    async def generate_response(
        self, prompt: str, context: Optional[str] = None, max_tokens: int = 500
    ):
        """Generate response with enhanced context handling"""
        try:
            print("\n[DEBUG] Generating response")
            print(f"[DEBUG] Prompt: {prompt[:200]}...")
            if context:
                print(f"[DEBUG] Context: {context[:200]}...")

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

            return await self.generate_response(prompt)

        except Exception as e:
            return {"status": "error", "response": None, "error": str(e)}

    async def generate_paper_comparison(
        self, papers: List[PaperMetadata]
    ) -> Dict[str, Any]:
        """Generate paper comparison"""
        try:
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

            return await self.generate_response(prompt, max_tokens=500)

        except Exception as e:
            return {"status": "error", "response": None, "error": str(e)}


class SearchAgent:
    """Handles paper search operations"""

    def __init__(self, s2_client: SemanticScholarClient):
        """Initialize search agent"""
        try:
            print("[DEBUG] Initializing SearchAgent")
            if not s2_client:
                raise ValueError("SemanticScholarClient cannot be None")
            self.client = s2_client
            print("[DEBUG] SearchAgent initialized successfully")
        except Exception as e:
            print(f"[DEBUG] Error initializing SearchAgent: {str(e)}")
            raise

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
                    "results": [],
                    "error": None,
                }

            return {"status": "success", "results": results, "error": None}

        except Exception as e:
            print(f"[DEBUG] SearchAgent: Error during search: {str(e)}")
            return {"status": "error", "results": None, "error": str(e)}


class EnhancedWorkflowManager:
    """Enhanced workflow manager that integrates the hierarchical system"""

    def __init__(self, model_name: str = "llama3.2:1b-instruct-q3_K_M"):
        """Initialize the workflow manager with hierarchical components"""
        try:
            print("[DEBUG] Initializing EnhancedWorkflowManager")

            # Initialize components
            self.state = AgentState()
            self.main_agent = MainAgent(model_name=model_name)
            self.workflow_graph = WorkflowGraph(model_name=model_name)

            print("[DEBUG] EnhancedWorkflowManager initialized successfully")

        except Exception as e:
            print(f"[DEBUG] Error in EnhancedWorkflowManager initialization: {str(e)}")
            raise

            # Initialize clients
            print("[DEBUG] Initializing clients...")
            self.ollama_client = OllamaClient(model_name=model_name)
            self.s2_client = SemanticScholarClient()
            print("[DEBUG] Clients initialized successfully")

            # Initialize tools and agents
            print("[DEBUG] Initializing tools and agents...")
            # Initialize state first
            self.current_state = AgentState()

            self.semantic_scholar_tool = SemanticScholarTool(state=self.current_state)
            self.paper_analyzer_tool = PaperAnalyzerTool(
                model_name=model_name, state=self.current_state
            )
            self.ollama_tool = OllamaTool(
                model_name=model_name, state=self.current_state
            )

            self.conversation_agent = ConversationAgent(
                ollama_client=self.ollama_client
            )
            self.search_agent = SearchAgent(s2_client=self.s2_client)
            print("[DEBUG] Tools and agents initialized successfully")

            # Initialize workflow manager
            print("[DEBUG] Initializing workflow manager...")
            self.workflow_manager = ResearchWorkflowManager(model_name=model_name)
            print("[DEBUG] Workflow manager initialized successfully")

            # Initialize state
            print("[DEBUG] Initializing state...")
            self.current_state = AgentState()
            print("[DEBUG] State initialized successfully")

            print("[DEBUG] EnhancedWorkflowManager initialization complete")

        except Exception as e:
            print(f"[DEBUG] Error in EnhancedWorkflowManager initialization: {str(e)}")
            raise

    async def process_command_async(self, command: str) -> AgentState:
        """Process commands using the hierarchical workflow"""
        try:
            print(f"\n[DEBUG] Processing command: {command}")

            # Update state with new command
            self.state.add_message("user", command)
            self.state.update_state(
                status=AgentStatus.PROCESSING,
                current_step="command_received",
                next_steps=["process_intent"],
            )

            # Process through workflow graph
            result = await self.workflow_graph.process_request(command)

            # Update final state
            self.state = result
            self.state.current_step = "completed"
            self.state.status = AgentStatus.SUCCESS

            return self.state

        except Exception as e:
            print(f"[DEBUG] Error in workflow: {str(e)}")
            self.state.status = AgentStatus.ERROR
            self.state.error_message = str(e)
            self.state.add_message(
                "system", f"I apologize, but I encountered an error: {str(e)}"
            )
            return self.state

    async def _handle_conversation(self, command: str):
        """Handle general conversation"""
        try:
            print(f"[DEBUG] Handling conversation command: {command}")

            if not hasattr(self, "conversation_agent"):
                print("[DEBUG] conversation_agent not found, reinitializing...")
                self.conversation_agent = ConversationAgent(
                    ollama_client=self.ollama_client
                )

            if "hi" in command.lower() or "hello" in command.lower():
                print("[DEBUG] Detected greeting")
                response = "Hello! I'm your research assistant. I can help you search for academic papers, analyze them, and answer questions about them. What would you like to know?"
            elif "what can you do" in command.lower():
                print("[DEBUG] Detected capabilities question")
                response = """I can help you with several tasks:
1. Search for academic papers on any topic
2. Analyze and explain specific papers
3. Compare different papers
4. Answer questions about research topics
5. Provide paper summaries and insights

What would you like me to help you with?"""
            else:
                print("[DEBUG] Processing general conversation")
                context = self._build_conversation_context()
                print(f"[DEBUG] Built context: {context[:100]}...")

                response_data = await self.conversation_agent.generate_response(
                    prompt=command, context=context
                )
                print(f"[DEBUG] Generated response data: {response_data}")

                if response_data["status"] == "error":
                    raise Exception(
                        f"Response generation failed: {response_data['error']}"
                    )
                response = response_data["response"]

            print("[DEBUG] Adding response to state")
            self.current_state.add_message("system", response)
            self.current_state.status = AgentStatus.SUCCESS

        except Exception as e:
            print(f"[DEBUG] Error in _handle_conversation: {str(e)}")
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            self.current_state.add_message(
                "system",
                f"I apologize, but I encountered an error while processing your message: {str(e)}",
            )

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
                    "abstract_preview": paper["abstract"][:200],
                }
                for paper in formatted_papers
            ]

            try:
                summary_prompt = f"""Based on these papers about "{clean_query}", please provide a brief summary of:
                    1. Main research areas covered
                    2. Key methodologies mentioned
                    3. Notable findings
                    4. Visible trends
                    
                    Papers: {json.dumps(summary_context)}"""

                summary_response = await self.conversation_agent.generate_response(
                    prompt=summary_prompt, max_tokens=300
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

            # Return state with next steps
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
            self.current_state.add_message(
                "system", f"I encountered an error while searching: {str(e)}"
            )
            return {"state": self.current_state, "next": "update_memory"}

    async def _handle_paper_question(self, paper_reference: str, question: str):
        """Handle paper-specific questions"""
        try:
            print(f"[DEBUG] Processing paper question for reference: {paper_reference}")

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
            print(f"[DEBUG] Error in paper question handling: {str(e)}")
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            self.current_state.add_message(
                "system",
                f"I apologize, but I encountered an error while analyzing the paper: {str(e)}",
            )

    async def _handle_paper_comparison(self, paper_references: List[str]):
        """Handle paper comparison requests"""
        try:
            print(
                f"[DEBUG] Processing paper comparison for references: {paper_references}"
            )

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
            print(f"[DEBUG] Error in paper comparison: {str(e)}")
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            self.current_state.add_message(
                "system",
                f"I apologize, but I encountered an error while comparing the papers: {str(e)}",
            )

    def _get_paper_by_reference(self, reference: str) -> Optional[PaperMetadata]:
        """Get paper by reference number"""
        try:
            if reference.isdigit():
                index = int(reference) - 1
                if 0 <= index < len(self.current_state.search_context.results):
                    return self.current_state.search_context.results[index]
            return None
        except Exception as e:
            print(f"[DEBUG] Error getting paper reference: {str(e)}")
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

    async def check_workflow_health(self) -> Dict[str, Any]:
        """Check health of all workflow components"""
        try:
            # Check main agent health
            agent_health = await self.main_agent.check_health()

            # Check workflow graph health
            workflow_health = True  # We'll add more specific checks later

            # Aggregate results
            return {
                "workflow_graph": workflow_health,
                "main_agent": agent_health.get("main_agent", False),
                "semantic_scholar": agent_health.get("semantic_scholar", False),
                "ollama_client": agent_health.get("ollama", False),
                "command_parser": True,
                "errors": [],
            }

        except Exception as e:
            return {
                "workflow_graph": False,
                "main_agent": False,
                "semantic_scholar": False,
                "ollama_client": False,
                "command_parser": False,
                "errors": [str(e)],
            }

    def _build_context(self) -> Dict[str, Any]:
        """Build current context for tools and agents"""
        return {
            "current_step": self.state.current_step,
            "conversation_history": (
                self.state.memory.messages[-5:] if self.state.memory.messages else []
            ),
            "current_papers": (
                self.state.search_context.results if self.state.search_context else []
            ),
            "focused_paper": (
                self.state.memory.focused_paper.paper_id
                if self.state.memory.focused_paper
                else None
            ),
        }
