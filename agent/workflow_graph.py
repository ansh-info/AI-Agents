import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")


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
        state.next_steps = ["analyze_input"]
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

    async def _handle_search(self, query: str):
        """Handle search with proper state management"""
        try:
            print(f"[DEBUG] Processing search for query: '{query}'")

            # Initialize state for search
            self.current_state.update_state(
                status=AgentStatus.PROCESSING,
                current_step="search_started",
                next_steps=["process_search"],
                last_update=datetime.now(),
            )

            # Perform search
            search_result = await self.search_agent.search_papers(query)

            if search_result["status"] == "error":
                raise Exception(f"Search failed: {search_result['error']}")

            results = search_result["results"]

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
                f"# Search Results\nFound {results.total} papers related to '{query}'.\n"
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
                    prompt=f"""Based on these papers about "{query}", please provide a brief summary of:
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
        """Build context for conversation"""
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
            if paper.abstract:
                context_parts.append(f"Abstract: {paper.abstract[:200]}...")
            context_parts.append(
                f"Authors: {', '.join(a.get('name', '') for a in paper.authors)}"
            )
            context_parts.append(
                f"Year: {paper.year or 'N/A'} | Citations: {paper.citations or 0}"
            )

        # Join all context parts
        return "\n".join(context_parts)

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
