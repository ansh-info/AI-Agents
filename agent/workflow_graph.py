import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from clients.ollama_client import OllamaClient
from clients.semantic_scholar_client import SemanticScholarClient
from state.agent_state import AgentState, AgentStatus, ConversationMemory


class WorkflowGraph:
    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        s2_client: Optional[SemanticScholarClient] = None,
    ):
        """Initialize the workflow graph with client integrations"""
        self.graph = StateGraph(AgentState)
        self.current_state = AgentState()
        self.ollama_client = ollama_client or OllamaClient()
        self.s2_client = s2_client or SemanticScholarClient()
        self.setup_graph()

    def setup_graph(self):
        """Setup the state graph with enhanced nodes and edges"""
        # Add core nodes
        self.graph.add_node("start", self._start_node)
        self.graph.add_node("analyze_input", self._analyze_input)
        self.graph.add_node("route", self._route_message)
        self.graph.add_node("process_search", self._process_search)
        self.graph.add_node("process_paper_question", self._process_paper_question)
        self.graph.add_node("process_conversation", self._process_conversation)
        self.graph.add_node("update_memory", self._update_memory)

        # Define the workflow edges
        self.graph.add_edge("start", "analyze_input")
        self.graph.add_edge("analyze_input", "route")
        self.graph.add_edge("route", "process_search")
        self.graph.add_edge("route", "process_paper_question")
        self.graph.add_edge("route", "process_conversation")
        self.graph.add_edge("process_search", "update_memory")
        self.graph.add_edge("process_paper_question", "update_memory")
        self.graph.add_edge("process_conversation", "update_memory")
        self.graph.add_edge("update_memory", END)

        # Set entry point
        self.graph.set_entry_point("start")

    def _start_node(self, state: AgentState) -> Dict:
        """Initialize state for new message processing"""
        state.status = AgentStatus.PROCESSING
        state.current_step = "start"
        return {"state": state, "next": "analyze_input"}

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
            state.error_message = f"Error in graph processing: {str(e)}"
            return state

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
            state.error_message = f"Error in input analysis: {str(e)}"
            return {
                "error": f"Error getting conversation context: {str(e)}",
                "current_step": state.current_step,
                "status": state.status.value,
            }
            return {"state": state, "next": END}

    def _route_message(self, state: AgentState) -> Dict:
        """Enhanced message routing with context awareness"""
        try:
            message = state.memory.messages[-1]["content"].lower()
            next_node = "process_conversation"

            # Check if we're in a paper context
            if state.memory.focused_paper:
                next_node = "process_paper_question"

            # Check for paper references
            elif state.search_context.results and any(
                ref in message
                for ref in [
                    "paper",
                    "study",
                    "research",
                    "article",
                    "tell me about",
                    "what is",
                    "compare",
                ]
            ):
                next_node = "process_paper_question"

            # Check for search intent
            elif any(
                term in message
                for term in ["search", "find", "look for", "papers about", "papers on"]
            ):
                next_node = "process_search"

            state.current_step = "routed"
            return {"state": state, "next": next_node}

        except Exception as e:
            state.status = AgentStatus.ERROR
            state.error_message = f"Error in message routing: {str(e)}"
            return {"state": state, "next": END}

    async def _process_search(self, state: AgentState) -> Dict:
        """Process search request with direct S2 integration"""
        try:
            # Extract search query from latest message
            latest_message = state.memory.messages[-1]["content"]
            query = self._clean_search_query(latest_message)

            # Perform search using S2 client
            results = await self.s2_client.search_papers(
                query=query, limit=10  # Configurable
            )

            # Update state with search results
            state.search_context.query = query
            state.search_context.total_results = results.total
            state.search_context.results = []

            for paper in results.papers:
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
                state.search_context.add_paper(paper_data)

            # Generate response using Ollama
            response = await self._generate_search_response(state)
            state.add_message("system", response)

            state.current_step = "search_processed"
            state.status = AgentStatus.SUCCESS

        except Exception as e:
            state.status = AgentStatus.ERROR
            state.error_message = f"Search processing error: {str(e)}"

        return {"state": state, "next": "update_memory"}

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

Question: {state.memory.messages[-1]['content']}

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
            # Build conversation context
            context = self._build_conversation_context(state)

            # Generate response using Ollama
            prompt = f"""Context:
{context}

Current query: {state.memory.messages[-1]['content']}
Please provide a helpful response based on the available context."""

            response = await self.ollama_client.generate(
                prompt=prompt,
                system_prompt="You are a helpful academic research assistant.",
                max_tokens=300,
            )

            state.add_message("system", response.strip())
            state.current_step = "conversation_processed"
            state.status = AgentStatus.SUCCESS

        except Exception as e:
            state.status = AgentStatus.ERROR
            state.error_message = f"Conversation processing error: {str(e)}"

        return {"state": state, "next": "update_memory"}

    def _build_conversation_context(self, state: AgentState) -> str:
        """Build context for conversation"""
        try:
            context_parts = []

            # Add recent conversation history
            recent_messages = state.memory.messages[-5:]
            if recent_messages:
                context_parts.append("Recent conversation:")
                for msg in recent_messages:
                    context_parts.append(f"{msg['role']}: {msg['content']}")

            # Add search context if available
            if state.search_context.results:
                context_parts.append("\nAvailable papers:")
                for i, paper in enumerate(state.search_context.results, 1):
                    context_parts.append(
                        f"{i}. {paper.title} ({paper.year or 'N/A'}) - {paper.citations or 0} citations"
                    )

            # Add focused paper if available
            if state.memory.focused_paper:
                paper = state.memory.focused_paper
                context_parts.append(f"\nCurrently discussing paper: {paper.title}")

            return "\n".join(context_parts)

        except Exception as e:
            state.status = AgentStatus.ERROR
            state.error_message = f"Context building error: {str(e)}"
            return ""

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
            state.status = AgentStatus.ERROR
            state.error_message = f"Response generation error: {str(e)}"
            return "I apologize, but I encountered an error generating the search response."

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
            if paper.title.lower() in message:
                return paper

        return None

    def _clean_search_query(self, query: str) -> str:
        """Clean search query"""
        clean_phrases = [
            "search for",
            "find me",
            "look for",
            "papers about",
            "papers on",
            "research on",
            "can you find",
        ]
        cleaned = query.lower()
        for phrase in clean_phrases:
            cleaned = cleaned.replace(phrase, "")
        return cleaned.strip()

    def _update_memory(self, state: AgentState) -> Dict:
        """Update conversation memory with enhanced context tracking"""
        try:
            # Update context history
            if hasattr(state.memory, "context_history"):
                context_entry = {
                    "timestamp": datetime.now(),
                    "step": state.current_step,
                    "focused_paper": (
                        state.memory.focused_paper.paper_id
                        if state.memory.focused_paper
                        else None
                    ),
                    "has_search_results": bool(state.search_context.results),
                }
                state.memory.context_history.append(context_entry)

            if state.status != AgentStatus.ERROR:
                state.status = AgentStatus.SUCCESS

            return {"state": state, "next": END}

        except Exception as e:
            state.status = AgentStatus.ERROR
            state.error_message = f"Error in memory update: {str(e)}"
            return {"state": state, "next": END}

    def get_graph(self):
        """Get the compiled graph"""
        return self.graph.compile()

    async def process_state(self, state: AgentState) -> AgentState:
        """Process a state through the graph"""
        try:
            graph_chain = self.get_graph()
            result = await graph_chain.ainvoke({"state": state})

            if isinstance(result, dict) and "state" in result:
                return result["state"]
            return result

        except Exception as e:
            state.status = AgentStatus.ERROR
