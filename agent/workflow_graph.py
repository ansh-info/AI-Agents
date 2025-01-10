import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from state.agent_state import AgentState, AgentStatus, ConversationMemory


class WorkflowGraph:
    def __init__(self):
        """Initialize the workflow graph with enhanced state management"""
        self.graph = StateGraph(AgentState)
        self.current_state = AgentState()
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
            state.error_message = f"Error in input analysis: {str(e)}"
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

    def _extract_paper_reference(
        self, message: str, state: AgentState
    ) -> Optional[Dict]:
        """Enhanced paper reference extraction"""
        if not state.search_context.results:
            return None

        # Check for numeric references (e.g., "paper 2")
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

    def _process_search(self, state: AgentState) -> Dict:
        """Process search request"""
        state.current_step = "search_processed"
        return {"state": state, "next": "update_memory"}

    def _process_paper_question(self, state: AgentState) -> Dict:
        """Process paper-related questions"""
        state.current_step = "paper_processed"
        return {"state": state, "next": "update_memory"}

    def _process_conversation(self, state: AgentState) -> Dict:
        """Process general conversation"""
        state.current_step = "conversation_processed"
        return {"state": state, "next": "update_memory"}

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

    def process_state(self, state: AgentState) -> AgentState:
        """Process a state through the graph"""
        try:
            graph_chain = self.get_graph()
            result = graph_chain.invoke({"state": state})

            if isinstance(result, dict) and "state" in result:
                return result["state"]
            return result

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

    def get_conversation_context(self, state: AgentState) -> Dict[str, Any]:
        """Get current conversation context"""
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
        }
        return context
