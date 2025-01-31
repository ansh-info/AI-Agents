import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from agent.main_agent import MainAgent
from clients.ollama_client import OllamaClient
from clients.semantic_scholar_client import SemanticScholarClient
from state.agent_state import AgentState, AgentStatus
from state.conversation_memory import ConversationMemory
from state.search_context import SearchContext
from tools.ollama_tool import OllamaTool
from tools.paper_analyzer_tool import PaperAnalyzerTool
from tools.semantic_scholar_tool import SemanticScholarTool


class ResearchTeam:
    """Research team responsible for paper search and initial analysis"""

    def __init__(self, state: AgentState):
        self.state = state
        self.semantic_scholar_tool = SemanticScholarTool(state=self.state)
        self.paper_analyzer_tool = PaperAnalyzerTool(state=self.state)

    async def search_papers(self, query: str) -> Dict:
        """Execute paper search"""
        return await self.semantic_scholar_tool._arun(query)

    async def analyze_paper(self, paper_id: str, request: str) -> Dict:
        """Analyze specific paper"""
        return await self.paper_analyzer_tool._arun(paper_id=paper_id, request=request)


class WorkflowGraph:
    """Enhanced workflow graph with hierarchical team structure"""

    def __init__(self, model_name: str = "llama3.2:1b-instruct-q3_K_M"):
        print("[DEBUG] Initializing WorkflowGraph")

        # Initialize state and teams
        self.state = AgentState()  # Initialize state first
        self.main_agent = MainAgent(model_name=model_name)
        self.research_team = ResearchTeam(state=self.state)

        # Initialize tools
        self.semantic_scholar_tool = SemanticScholarTool(state=self.state)
        self.paper_analyzer_tool = PaperAnalyzerTool(
            model_name=model_name, state=self.state
        )
        self.ollama_tool = OllamaTool(model_name=model_name, state=self.state)

        # Create graph
        self.graph = self._create_workflow_graph()
        print("[DEBUG] WorkflowGraph initialized")

    def _create_workflow_graph(self) -> StateGraph:
        """Create the hierarchical workflow graph"""
        workflow = StateGraph(MessagesState)

        # Add nodes
        workflow.add_node("supervisor", self._main_agent_node)
        workflow.add_node("research_team", self._research_team_node)
        workflow.add_node("update_state", self._update_state_node)

        # Add edges
        workflow.add_edge(START, "supervisor")
        workflow.add_edge("supervisor", "research_team")
        workflow.add_edge("research_team", "update_state")
        workflow.add_edge("update_state", END)

        # Set the entry point
        self.graph = workflow.compile()
        return self.graph

    def setup_graph(self):
        """Setup the state graph with enhanced nodes and edges"""
        # Add core nodes
        self.graph.add_node("start", self._start_node)
        self.graph.add_node("analyze_intent", self._analyze_intent)
        self.graph.add_node("route_request", self._route_request)
        self.graph.add_node("search_papers", self._search_papers)
        self.graph.add_node("analyze_paper", self._analyze_paper)
        self.graph.add_node("handle_conversation", self._handle_conversation)
        self.graph.add_node("update_state", self._update_state)

        # Define edges
        self.graph.add_edge("start", "analyze_intent")
        self.graph.add_edge("analyze_intent", "route_request")
        self.graph.add_edge("route_request", "search_papers")
        self.graph.add_edge("route_request", "analyze_paper")
        self.graph.add_edge("route_request", "handle_conversation")
        self.graph.add_edge("search_papers", "update_state")
        self.graph.add_edge("analyze_paper", "update_state")
        self.graph.add_edge("handle_conversation", "update_state")
        self.graph.add_edge("update_state", END)

        # Set entry point
        self.graph.set_entry_point("start")

    async def _start_node(self, state: MessagesState) -> Dict:
        """Initialize request processing"""
        try:
            print("[DEBUG] Starting new request")
            self.state.update_state(
                status=AgentStatus.PROCESSING,
                current_step="start",
                next_steps=["main_agent"],
            )
            return {"messages": state["messages"], "next": "main_agent"}
        except Exception as e:
            print(f"[DEBUG] Error in start node: {str(e)}")
            self.state.status = AgentStatus.ERROR
            self.state.error_message = str(e)
            return {"state": self.state, "next": "update_state"}

    async def _main_agent_node(self, state: MessagesState) -> Dict:
        """Main agent processing"""
        try:
            message = state["messages"][-1]
            message_content = (
                message.content
                if isinstance(message, HumanMessage)
                else message["content"]
            )

            intent = await self.main_agent._determine_intent(message_content)

            if intent == "search":
                return {"messages": state["messages"], "next": "research_team"}
            else:
                result = await self.main_agent.process_request(message_content)
                return {
                    "messages": [
                        *state["messages"],
                        AIMessage(content=result.memory.messages[-1]["content"]),
                    ],
                    "next": "update_state",
                }
        except Exception as e:
            print(f"[DEBUG] Error in main agent: {str(e)}")
            return {
                "messages": [
                    *state["messages"],
                    SystemMessage(content=f"Error: {str(e)}"),
                ],
                "next": "update_state",
            }

    async def _research_team_node(self, state: MessagesState) -> Dict:
        try:
            # Extract content safely from last message
            last_message = state["messages"][-1]
            message_content = (
                last_message.content
                if isinstance(last_message, HumanMessage)
                else last_message["content"]
                if isinstance(last_message, dict)
                else str(last_message)
            )

            result = await self.research_team.search_papers(message_content)

            # Update state with search results if available
            if isinstance(result, dict) and "papers" in result:
                self.state.search_context.results = result["papers"]

            # Format response
            response = self.main_agent._format_search_results(result)

            # Return updated state with new message
            return {
                "messages": [
                    *state["messages"],
                    {"role": "system", "content": response},
                ],
                "next": "update_state",
            }
        except Exception as e:
            print(f"[DEBUG] Error in research team: {str(e)}")
            return {
                "messages": [
                    *state["messages"],
                    {"role": "system", "content": f"Error in search: {str(e)}"},
                ],
                "next": "update_state",
            }

    async def _update_state_node(self, state: MessagesState) -> Dict:
        """Update final state"""
        self.state.last_update = datetime.now()
        if not hasattr(self.state, "state_history"):
            self.state.state_history = []

        self.state.state_history.append(
            {
                "timestamp": datetime.now(),
                "step": self.state.current_step,
                "status": self.state.status.value,
            }
        )

        return {"messages": state["messages"], "next": END}

    async def _analyze_intent(self, state: AgentState) -> Dict:
        """Analyze request intent using Ollama"""
        try:
            print("[DEBUG] Analyzing request intent")

            # Get latest message
            if not state.memory.messages:
                raise ValueError("No messages in state")

            message = state.memory.messages[-1]["content"]

            # Use Ollama to determine intent
            intent_prompt = f"""Analyze this user message and determine the intent:
Message: "{message}"

Possible intents:
1. search - User wants to find papers
2. analyze - User wants to analyze specific papers
3. conversation - General questions or chat

Return only one word (search/analyze/conversation)."""

            intent = await self.ollama_tool._arun(
                prompt=intent_prompt,
                system_prompt="You are an intent classifier. Return only one word.",
                temperature=0.1,
            )

            # Update state
            state.update_state(
                current_step="intent_analyzed", next_steps=["route_request"]
            )
            state.memory.current_context = intent.strip().lower()

            return {"state": state, "next": "route_request"}

        except Exception as e:
            print(f"[DEBUG] Error analyzing intent: {str(e)}")
            state.status = AgentStatus.ERROR
            state.error_message = str(e)
            return {"state": state, "next": "update_state"}

    async def _route_request(self, state: AgentState) -> Dict:
        """Route request to appropriate handler based on intent"""
        try:
            intent = state.memory.current_context
            print(f"[DEBUG] Routing request with intent: {intent}")

            # Map intent to next node
            intent_map = {
                "search": "search_papers",
                "analyze": "analyze_paper",
                "conversation": "handle_conversation",
            }

            next_node = intent_map.get(intent, "handle_conversation")
            state.update_state(current_step="routed", next_steps=[next_node])

            return {"state": state, "next": next_node}

        except Exception as e:
            print(f"[DEBUG] Error routing request: {str(e)}")
            state.status = AgentStatus.ERROR
            state.error_message = str(e)
            return {"state": state, "next": "update_state"}

    async def _search_papers(self, state: AgentState) -> Dict:
        """Handle paper search requests"""
        try:
            print("[DEBUG] Processing search request")
            message = state.memory.messages[-1]["content"]

            # Execute search
            result = await self.semantic_scholar_tool._arun(message)

            # Update state
            state.add_message("system", result)
            state.update_state(
                current_step="search_completed",
                next_steps=["update_state"],
                status=AgentStatus.SUCCESS,
            )

            return {"state": state, "next": "update_state"}

        except Exception as e:
            print(f"[DEBUG] Error in search: {str(e)}")
            state.status = AgentStatus.ERROR
            state.error_message = str(e)
            return {"state": state, "next": "update_state"}

    async def _analyze_paper(self, state: AgentState) -> Dict:
        """Handle paper analysis requests"""
        try:
            print("[DEBUG] Processing paper analysis request")
            message = state.memory.messages[-1]["content"]

            # Execute analysis
            result = await self.paper_analyzer_tool._arun(message)

            # Update state
            state.add_message("system", result)
            state.update_state(
                current_step="analysis_completed",
                next_steps=["update_state"],
                status=AgentStatus.SUCCESS,
            )

            return {"state": state, "next": "update_state"}

        except Exception as e:
            print(f"[DEBUG] Error in paper analysis: {str(e)}")
            state.status = AgentStatus.ERROR
            state.error_message = str(e)
            return {"state": state, "next": "update_state"}

    async def _handle_conversation(self, state: AgentState) -> Dict:
        """Handle general conversation"""
        try:
            print("[DEBUG] Processing conversation")
            message = state.memory.messages[-1]["content"]

            # Generate response
            result = await self.ollama_tool._arun(message)

            # Update state
            state.add_message("system", result)
            state.update_state(
                current_step="conversation_completed",
                next_steps=["update_state"],
                status=AgentStatus.SUCCESS,
            )

            return {"state": state, "next": "update_state"}

        except Exception as e:
            print(f"[DEBUG] Error in conversation: {str(e)}")
            state.status = AgentStatus.ERROR
            state.error_message = str(e)
            return {"state": state, "next": "update_state"}

    async def _update_state(self, state: AgentState) -> Dict:
        """Update final state after processing"""
        try:
            print("[DEBUG] Updating final state")

            # Ensure all fields are properly set
            state.last_update = datetime.now()

            # Add state history entry
            if not hasattr(state, "state_history"):
                state.state_history = []

            state.state_history.append(
                {
                    "timestamp": datetime.now(),
                    "step": state.current_step,
                    "status": state.status.value,
                }
            )

            return {"state": state, "next": END}

        except Exception as e:
            print(f"[DEBUG] Error updating state: {str(e)}")
            state.status = AgentStatus.ERROR
            state.error_message = str(e)
            return {"state": state, "next": END}

    def _analyze_input(self, state: AgentState) -> Dict:
        """Analyze input message and extract context"""
        try:
            # Get the latest message
            if not state.memory.messages:
                return {"state": state, "next": END}

            latest_message = state.memory.messages[-1]
            message_content = latest_message["content"].lower()

            # Extract paper references if any
            paper_reference = self._extract_paper_reference(message_content, state)
            if paper_reference:
                state.memory.current_context = "paper_reference"
                if isinstance(paper_reference, dict):
                    state.memory.focused_paper = paper_reference

            # Update state with analysis
            state.current_step = "analyzed"
            return {"state": state, "next": "route"}

        except Exception as e:
            state.status = AgentStatus.ERROR
            state.error_message = f"Error in input analysis: {str(e)}"
            return {"state": state, "next": END}

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

            # Check health
            health = await self.check_clients_health()
            return health["all_healthy"]
        except Exception:
            return False

    def get_conversation_context(self, state: AgentState) -> Dict[str, Any]:
        """Get current conversation context"""
        try:
            context = {
                "current_step": state.current_step,
                "has_search_results": bool(state.search_context.results),
                "focused_paper": (
                    state.memory.focused_paper.paper_id
                    if state.memory.focused_paper
                    else None
                ),
                "recent_messages": (
                    state.memory.messages[-5:] if state.memory.messages else []
                ),
                "status": state.status.value,
                "search_context": {
                    "query": state.search_context.query,
                    "total_results": state.search_context.total_results,
                    "current_page": state.search_context.current_page,
                },
            }
            return context
        except Exception as e:
            return {
                "error": f"Error getting conversation context: {str(e)}",
                "current_step": state.current_step,
                "status": state.status.value,
            }

    def _route_message(self, state: AgentState) -> Dict:
        """Enhanced message routing with context awareness"""
        try:
            message = state.memory.messages[-1]["content"].lower()
            next_node = "process_conversation"

            # Check if we're in a paper context
            if state.memory.focused_paper:
                next_node = "process_paper_question"

            # Check for search intent
            elif any(
                term in message
                for term in [
                    "search",
                    "find",
                    "look for",
                    "papers about",
                    "papers on",
                    "fetch",
                ]
            ):
                next_node = "process_search"

            # Update state with required fields
            state.current_step = "routed"
            state.last_update = datetime.now()
            state.next_steps = [next_node]
            if not hasattr(state, "state_history"):
                state.state_history = []
            state.state_history.append(
                {"timestamp": datetime.now(), "step": "routing", "next_node": next_node}
            )

            return {"state": state, "next": next_node}

        except Exception as e:
            state.status = AgentStatus.ERROR
            state.error_message = f"Error in message routing: {str(e)}"
            state.current_step = "error"
            state.last_update = datetime.now()
            state.next_steps = ["update_memory"]
            if not hasattr(state, "state_history"):
                state.state_history = []
            state.state_history.append(
                {"timestamp": datetime.now(), "step": "error", "error": str(e)}
            )
            return {"state": state, "next": "update_memory"}

    def _process_search(self, state: AgentState) -> Dict:
        """Process search operation with proper state updates"""
        try:
            # Initialize state for search
            state.update_state(
                status=AgentStatus.PROCESSING,
                current_step="search_started",
                next_steps=["update_memory"],
                last_update=datetime.now(),
            )

            # Ensure state_history exists and is updated
            if not hasattr(state, "state_history"):
                state.state_history = []

            state.state_history.append(
                {
                    "timestamp": datetime.now(),
                    "step": "search_started",
                    "query": state.search_context.query if state.search_context else "",
                }
            )

            # Ensure all required fields are set
            if not state.error_message:
                state.error_message = None
            if not state.memory:
                state.memory = ConversationMemory()
            if not state.search_context:
                state.search_context = SearchContext()

            return {"state": state, "next": "update_memory"}
        except Exception as e:
            print(f"[DEBUG] Error in search processing: {str(e)}")
            state.update_state(
                status=AgentStatus.ERROR,
                error_message=str(e),
                current_step="error",
                next_steps=["update_memory"],
                last_update=datetime.now(),
            )
            if not hasattr(state, "state_history"):
                state.state_history = []
            state.state_history.append(
                {"timestamp": datetime.now(), "step": "error", "error": str(e)}
            )
            return {"state": state, "next": "update_memory"}

    async def _generate_summary(self, query: str, papers: List[Dict]) -> str:
        """Generate summary of search results"""
        try:
            # Create a safe context for the LLM
            papers_info = []
            for paper in papers[:10]:  # Limit to first 10 papers
                papers_info.append(
                    {
                        "title": paper.get("title", ""),
                        "year": paper.get("year", "N/A"),
                        "citations": paper.get("citations", 0),
                        "abstract_preview": (
                            paper.get("abstract", "")[:200]
                            if paper.get("abstract")
                            else "No abstract available"
                        ),
                    }
                )

            summary_prompt = f"""Based on these papers about "{query}", please provide a brief summary of:
    1. Main research themes found
    2. Key methodologies mentioned
    3. Notable findings or contributions
    4. Any visible trends across the papers

    Papers: {json.dumps(papers_info, indent=2)}"""

            # Generate summary
            response = await self.conversation_agent.generate_response(
                prompt=summary_prompt,
                system_prompt="You are a helpful academic research assistant. Focus on summarizing the specific papers provided.",
                max_tokens=300,
            )

            if response["status"] == "success":
                return response["response"]
            else:
                return "Unable to generate summary - error in response generation"

        except Exception as e:
            print(f"[DEBUG] Summary generation error: {str(e)}")
            return "Unable to generate summary at this time."

    async def _handle_search(self, request: str) -> AgentState:
        """Handle search requests with proper state management"""
        try:
            print(f"[DEBUG] Handling search request: {request}")

            # Extract search parameters
            search_params = self._extract_search_params(request)
            print(f"[DEBUG] Extracted search params: {search_params}")

            # Execute search through semantic scholar tool
            search_result = await self.semantic_scholar_tool._arun(
                query=search_params.get("query", request),
                year_start=search_params.get("year_start"),
                year_end=search_params.get("year_end"),
                min_citations=search_params.get("min_citations"),
            )

            # Check for error in search result
            if (
                isinstance(search_result, dict)
                and search_result.get("status") == "error"
            ):
                raise Exception(f"Search failed: {search_result.get('error')}")

            # Update state with search results
            if isinstance(search_result, dict) and "papers" in search_result:
                # Convert papers to PaperContext objects
                paper_contexts = []
                for paper in search_result["papers"]:
                    paper_ctx = PaperContext(
                        paperId=paper["id"],
                        title=paper["title"],
                        authors=[{"name": name} for name in paper["authors"]],
                        year=paper.get("year"),
                        citations=paper.get("citations", 0),
                        abstract=paper.get("abstract", ""),
                        url=paper.get("url", ""),
                    )
                    paper_contexts.append(paper_ctx)

                # Update state
                self.state.search_context.results = paper_contexts
                self.state.search_context.query = request

                # Format and add response
                response = self._format_search_results(search_result)
                self.state.add_message("system", response)
                print(
                    f"[DEBUG] Search successful, found {len(search_result['papers'])} papers"
                )

            # Update state status
            self.state.status = AgentStatus.SUCCESS
            self.state.current_step = "search_completed"

            return self.state

        except Exception as e:
            print(f"[DEBUG] Error in search handler: {str(e)}")
            self.state.status = AgentStatus.ERROR
            self.state.error_message = str(e)
            self.state.current_step = "search_failed"
            return self.state

    def _extract_search_params(self, request: str) -> Dict[str, Any]:
        """Extract search parameters from request"""
        params = {
            "query": request,
            "year_start": None,
            "year_end": None,
            "min_citations": None,
        }

        # Extract year range
        year_patterns = [
            (r"from\s+(\d{4})", "year_start"),
            (r"since\s+(\d{4})", "year_start"),
            (r"after\s+(\d{4})", "year_start"),
            (r"before\s+(\d{4})", "year_end"),
            (r"until\s+(\d{4})", "year_end"),
            (r"in\s+(\d{4})", "year_exact"),
        ]

        for pattern, param in year_patterns:
            match = re.search(pattern, request, re.IGNORECASE)
            if match:
                year = int(match.group(1))
                if param == "year_exact":
                    params["year_start"] = year
                    params["year_end"] = year
                else:
                    params[param] = year

        # Extract citation threshold
        citation_match = re.search(
            r"at least\s+(\d+)\s+citations", request, re.IGNORECASE
        )
        if citation_match:
            params["min_citations"] = int(citation_match.group(1))

        return params

    def _format_search_results(self, results: Dict[str, Any]) -> str:
        """Format search results for display"""
        if results.get("status") == "error":
            return f"Error performing search: {results.get('error')}"

        papers = results.get("papers", [])
        if not papers:
            return "No papers found matching your criteria."

        formatted_parts = [f"Found {len(papers)} papers matching your criteria:\n"]

        for i, paper in enumerate(papers, 1):
            paper_info = [
                f"\n{i}. {paper.get('title', 'Untitled')}",
                f"Authors: {', '.join(paper.get('authors', []))}",
                f"Year: {paper.get('year', 'N/A')} | Citations: {paper.get('citations', 0)}",
            ]

            if paper.get("abstract"):
                abstract = paper["abstract"]
                paper_info.append(
                    f"Abstract: {abstract[:300]}..."
                    if len(abstract) > 300
                    else f"Abstract: {abstract}"
                )

            if paper.get("url"):
                paper_info.append(f"URL: {paper.get('url')}\n")

            formatted_parts.extend(paper_info)

        return "\n".join(formatted_parts)

    async def _process_paper_question(self, state: AgentState) -> Dict:
        """Process paper-related questions with LLM integration"""
        try:
            # Get the paper being discussed
            paper = state.memory.focused_paper or self._extract_paper_reference(
                state.memory.messages[-1]["content"], state
            )

            if not paper:
                response = "I'm not sure which paper you're referring to. Could you specify the paper number or title?"
                state.add_message("system", response)
                return {"state": state, "next": "update_memory"}

            # Build context for the LLM
            paper_context = (
                f"Title: {paper.title}\n"
                f"Authors: {', '.join(a.get('name', '') for a in paper.authors)}\n"
                f"Year: {paper.year or 'Not specified'}\n"
                f"Citations: {paper.citations or 0}\n"
                f"Abstract: {paper.abstract or 'Not available'}\n"
            )

            # Generate response using Ollama
            prompt = f"""Based on this academic paper:
{paper_context}

Question: {state.memory.messages[-1]["content"]}

Please provide a clear response that addresses the question while considering:
1. The paper's main findings
2. Relevant methodology
3. Key conclusions
4. Any limitations mentioned
5. The broader impact of the research"""

            response = await self.ollama_client.generate(
                prompt=prompt,
                system_prompt="You are a helpful academic research assistant.",
                max_tokens=300,
            )

            state.memory.focused_paper = paper
            state.add_message("system", response.strip())
            state.current_step = "paper_processed"
            state.status = AgentStatus.SUCCESS

        except Exception as e:
            state.status = AgentStatus.ERROR
            state.error_message = f"Paper question processing error: {str(e)}"

        return {"state": state, "next": "update_memory"}

    async def _process_conversation(self, state: AgentState) -> Dict:
        """Process general conversation with context-aware LLM integration"""
        try:
            # Initialize state
            state.update_state(
                current_step="conversation_processing",
                next_steps=["update_memory"],
                status=AgentStatus.PROCESSING,
            )

            # Build conversation context
            context = self._build_conversation_context(state)

            # Generate response using Ollama
            prompt = f"Context:\n{context}\n\nCurrent query: {state.memory.messages[-1]['content']}"

            response = await self.conversation_agent.generate_response(
                prompt=prompt,
                system_prompt="You are a helpful academic research assistant.",
                max_tokens=300,
            )

            if response["status"] == "error":
                raise Exception(
                    f"Response generation failed: {response.get('error', 'Unknown error')}"
                )

            # Add response to state
            state.add_message("system", response["response"].strip())

            # Update state on success
            state.update_state(
                current_step="conversation_completed",
                status=AgentStatus.SUCCESS,
                next_steps=["update_memory"],
            )

            return {"state": state, "next": "update_memory"}

        except Exception as e:
            print(f"[DEBUG] Conversation processing error: {str(e)}")
            state.update_state(
                status=AgentStatus.ERROR,
                error_message=f"Conversation processing error: {str(e)}",
                current_step="error",
                next_steps=["update_memory"],
            )
            state.add_message(
                "system", f"I apologize, but I encountered an error: {str(e)}"
            )
            return {"state": state, "next": "update_memory"}

    def _build_conversation_context(self, state: AgentState) -> str:
        """Build context from conversation history and current state"""
        context_parts = []

        # Add recent conversation history (last 5 messages)
        if state.memory and state.memory.messages:
            recent_messages = state.memory.messages[-5:]
            context_parts.append("Recent conversation:")
            for msg in recent_messages:
                context_parts.append(f"{msg['role']}: {msg['content']}")

        # Add search context if available
        if state.search_context and state.search_context.results:
            context_parts.append("\nCurrent search results:")
            for i, paper in enumerate(state.search_context.results[:5], 1):
                context_parts.append(
                    f"{i}. {paper.title} ({paper.year or 'N/A'}) - {paper.citations or 0} citations"
                )

        # Add focused paper if available
        if state.memory.focused_paper:
            paper = state.memory.focused_paper
            context_parts.extend(
                [
                    "\nCurrently discussing paper:",
                    f"Title: {paper.title}",
                    f"Authors: {', '.join(a.get('name', '') for a in paper.authors)}",
                    f"Year: {paper.year or 'N/A'} | Citations: {paper.citations or 0}",
                    f"Abstract: {paper.abstract[:200] + '...' if paper.abstract else 'Not available'}",
                ]
            )

        return "\n".join(context_parts)

    async def _handle_history_query(self, request: str) -> Dict[str, Any]:
        """Handle queries about conversation history with contextual responses"""
        try:
            print("[DEBUG] Handling history query")

            # Build context including last search and focused paper
            context = self._build_conversation_context(self.state)

            # Add specific context based on the type of history query
            if "last search" in request.lower() and self.state.search_context.query:
                context += f"\nLast search query: {self.state.search_context.query}"

            if "paper" in request.lower():
                if self.state.memory.focused_paper:
                    paper = self.state.memory.focused_paper
                    context += f"\nCurrently focused paper: {paper.title} ({paper.year}) by {', '.join(a['name'] for a in paper.authors)}"
                elif self.state.search_context.results:
                    papers = self.state.search_context.results[:3]
                    context += "\nRecent papers found:\n" + "\n".join(
                        f"{i + 1}. {p.title}" for i, p in enumerate(papers)
                    )

            # Generate response using conversation agent
            response = await self.conversation_agent.generate_response(
                prompt=f"Based on the following context:\n{context}\n\nUser question: {request}",
                system_prompt="""You are a helpful research assistant. Use the conversation history to answer the user's question.
                    When referring to papers, include their titles and authors.
                    If discussing search results, mention the search query used.""",
            )

            if isinstance(response, dict) and response.get("status") == "error":
                raise Exception(f"Error generating response: {response.get('error')}")

            # Update state
            self.state.add_message("system", response.get("response", response))
            return {"status": AgentStatus.SUCCESS}

        except Exception as e:
            print(f"[DEBUG] Error handling history query: {str(e)}")
            return {"status": AgentStatus.ERROR, "error_message": str(e)}

    async def _generate_search_response(self, state: AgentState) -> str:
        """Generate structured search response using LLM"""
        try:
            context = f"Found {state.search_context.total_results} papers. Here are the top {len(state.search_context.results)} most relevant ones:\n\n"

            for i, paper in enumerate(state.search_context.results, 1):
                context += (
                    f"Paper {i}:\n"
                    f"Title: {paper.title}\n"
                    f"Authors: {', '.join(a.get('name', '') for a in paper.authors)}\n"
                    f"Year: {paper.year or 'N/A'}\n"
                    f"Citations: {paper.citations or 0}\n"
                )
                if paper.abstract:
                    context += f"Abstract: {paper.abstract[:200]}...\n"
                context += "---\n"

            prompt = f"""Based on these search results:
{context}

Please provide a structured response that:
1. Summarizes the search results
2. Highlights key papers
3. Notes any interesting patterns or trends
4. Suggests how to interact with these results"""

            response = await self.ollama_client.generate(
                prompt=prompt,
                system_prompt="You are a helpful academic research assistant.",
                max_tokens=300,
            )

            return response.strip()

        except Exception as e:
            return f"Error generating search response: {str(e)}"

    def _extract_paper_reference(
        self, message: str, state: AgentState
    ) -> Optional[Dict]:
        """Extract paper reference from message"""
        if not state.search_context.results:
            return None

        words = message.split()
        for i, word in enumerate(words):
            if word.isdigit():
                try:
                    paper_num = int(word)
                    if 1 <= paper_num <= len(state.search_context.results):
                        return state.search_context.get_paper_by_index(paper_num)
                except ValueError:
                    continue
            elif word in ["paper", "study", "article"] and i > 0:
                try:
                    prev_word = words[i - 1]
                    if prev_word.isdigit():
                        paper_num = int(prev_word)
                        if 1 <= paper_num <= len(state.search_context.results):
                            return state.search_context.get_paper_by_index(paper_num)
                except ValueError:
                    continue

        # Check for title references
        for paper in state.search_context.results:
            if paper.title.lower() in message.lower():
                return paper

        return None

    def _update_memory(self, state: AgentState) -> Dict:
        """Update conversation memory with enhanced context tracking"""
        try:
            # Ensure all required fields are updated
            state.update_state(
                current_step="memory_updated",
                next_steps=[],  # Clear next steps as this is the end
                last_update=datetime.now(),
                status=state.status,  # Preserve existing status
            )

            # Ensure state_history exists
            if not hasattr(state, "state_history"):
                state.state_history = []

            # Add state history entry
            state.state_history.append(
                {
                    "timestamp": datetime.now(),
                    "step": state.current_step,
                    "status": state.status.value,
                    "message_count": (
                        len(state.memory.messages) if state.memory.messages else 0
                    ),
                }
            )

            return {"state": state, "next": END}

        except Exception as e:
            print(f"[DEBUG] Error in memory update: {str(e)}")
            state.update_state(
                status=AgentStatus.ERROR,
                error_message=f"Error in memory update: {str(e)}",
                current_step="error",
                next_steps=[],
                last_update=datetime.now(),
            )

            # Always add history entry even in error case
            if not hasattr(state, "state_history"):
                state.state_history = []
            state.state_history.append(
                {
                    "timestamp": datetime.now(),
                    "step": "error",
                    "status": "error",
                    "error": str(e),
                }
            )

            return {"state": state, "next": END}

    def get_graph(self):
        """Get the compiled graph"""
        return self.graph.compile()

    async def debug_state(self, state: AgentState) -> Dict[str, Any]:
        """Debug current state of the workflow"""
        debug_info = {
            "current_step": state.current_step,
            "status": state.status.value,
            "error_message": state.error_message,
            "search_context": {
                "query": state.search_context.query,
                "total_results": state.search_context.total_results,
                "current_page": state.search_context.current_page,
                "num_results": (
                    len(state.search_context.results)
                    if state.search_context.results
                    else 0
                ),
            },
            "memory": {
                "num_messages": (
                    len(state.memory.messages) if state.memory.messages else 0
                ),
                "current_context": state.memory.current_context,
                "focused_paper": (
                    state.memory.focused_paper.paper_id
                    if state.memory.focused_paper
                    else None
                ),
                "conversation_topics": (
                    list(state.memory.conversation_topics)
                    if state.memory.conversation_topics
                    else []
                ),
            },
            "next_steps": state.next_steps,
        }

        print("[DEBUG] Current State Info:")
        print(json.dumps(debug_info, indent=2))
        return debug_info

    async def process_state(self, state: AgentState) -> AgentState:
        """Process state through the graph with enhanced state management"""
        try:
            if not hasattr(state, "current_step"):
                state.current_step = "initial"
            if not hasattr(state, "next_steps"):
                state.next_steps = []
            if not hasattr(state, "state_history"):
                state.state_history = []

            # Always update last_update when processing state
            state.last_update = datetime.now()

            # Add history entry for state processing start
            state.state_history.append(
                {
                    "timestamp": datetime.now(),
                    "step": state.current_step,
                    "status": state.status.value,
                }
            )

            # Process through graph
            graph_chain = self.get_graph()
            result = await graph_chain.ainvoke({"state": state})

            # Extract final state from result
            final_state = result.get("state") if isinstance(result, dict) else result

            # Ensure all required fields are updated
            final_state.last_update = datetime.now()
            final_state.current_step = final_state.current_step or "completed"
            final_state.next_steps = getattr(final_state, "next_steps", [])
            final_state.state_history = getattr(final_state, "state_history", [])

            # Add final history entry
            final_state.state_history.append(
                {
                    "timestamp": datetime.now(),
                    "step": final_state.current_step,
                    "status": final_state.status.value,
                }
            )

            return final_state

        except Exception as e:
            print(f"[DEBUG] Error processing state: {str(e)}")
            state.status = AgentStatus.ERROR
            state.error_message = str(e)
            state.current_step = "error"
            state.last_update = datetime.now()
            state.state_history = getattr(state, "state_history", [])
            state.state_history.append(
                {
                    "timestamp": datetime.now(),
                    "step": "error",
                    "status": "error",
                    "error": str(e),
                }
            )
            return state

    async def process_request(self, request: str) -> AgentState:
        """Process a request through the workflow with enhanced state management"""
        try:
            print(f"[DEBUG] Processing request: {request}")

            # Update state with new request
            self.state.add_message("user", request)

            # Build conversation context
            conversation_context = self._build_conversation_context(self.state)

            # Determine intent with context
            intent = await self.main_agent._determine_intent(
                request,
                context={
                    "conversation_history": conversation_context,
                    "current_search_results": bool(self.state.search_context.results),
                    "focused_paper": self.state.memory.focused_paper.paper_id
                    if self.state.memory.focused_paper
                    else None,
                },
            )
            print(f"[DEBUG] Intent analysis result: {intent}")

            # Check for search keywords
            if any(
                keyword in request.lower()
                for keyword in [
                    "find",
                    "search",
                    "papers about",
                    "papers on",
                    "papers by",
                ]
            ):
                return await self._handle_search(request)

            # Process based on intent
            if intent.get("intent") == "conversation":
                if self._is_history_query(request):
                    await self._handle_history_query(request)
                else:
                    await self._handle_conversation(request)
            else:  # search intent
                await self._handle_search(request)

            self.state.status = AgentStatus.SUCCESS
            return self.state

        except Exception as e:
            print(f"[DEBUG] Error processing request: {str(e)}")
            self.state.status = AgentStatus.ERROR
            self.state.error_message = str(e)
            self.state.add_message(
                "system", f"I apologize, but I encountered an error: {str(e)}"
            )
            return self.state

    def _is_history_query(self, request: str) -> bool:
        """Check if the request is asking about conversation history"""
        history_keywords = [
            "previous",
            "last",
            "before",
            "earlier",
            "recent",
            "what did you say",
            "what did we discuss",
            "what papers",
            "what was the",
        ]
        request_lower = request.lower()
        return any(keyword in request_lower for keyword in history_keywords)

    async def check_health(self) -> Dict[str, bool]:
        """Check health of all components"""
        try:
            semantic_scholar_health = await self.semantic_scholar_tool.check_health()
            paper_analyzer_health = await self.paper_analyzer_tool.check_health()
            ollama_health = await self.ollama_tool.check_health()

            return {
                "semantic_scholar": semantic_scholar_health,
                "paper_analyzer": paper_analyzer_health,
                "ollama": ollama_health,
                "graph": True,
                "all_healthy": all(
                    [semantic_scholar_health, paper_analyzer_health, ollama_health]
                ),
            }
        except Exception as e:
            return {
                "semantic_scholar": False,
                "paper_analyzer": False,
                "ollama": False,
                "graph": False,
                "all_healthy": False,
                "error": str(e),
            }
