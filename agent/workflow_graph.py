import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

from state.agent_state import AgentState, AgentStatus, ConversationMemory


class WorkflowGraph:
    def __init__(self):
        """Initialize the workflow graph"""
        self.graph = StateGraph(AgentState)
        self.setup_graph()

    def setup_graph(self):
        """Setup the state graph with nodes and edges"""
        # Add nodes for different states
        self.graph.add_node("start", self._start_node)
        self.graph.add_node("route", self._route_message)
        self.graph.add_node("process_search", self._process_search)
        self.graph.add_node("process_paper_question", self._process_paper_question)
        self.graph.add_node("process_conversation", self._process_conversation)
        self.graph.add_node("update_memory", self._update_memory)

        # Define the edges
        self.graph.add_edge("start", "route")
        self.graph.add_edge("route", "process_search")
        self.graph.add_edge("route", "process_paper_question")
        self.graph.add_edge("route", "process_conversation")
        self.graph.add_edge("process_search", "update_memory")
        self.graph.add_edge("process_paper_question", "update_memory")
        self.graph.add_edge("process_conversation", "update_memory")
        self.graph.add_edge("update_memory", END)

        # Set entry point
        self.graph.set_entry_point("start")

    def _start_node(self, state: AgentState) -> Dict[str, Any]:
        """Initialize state for new message"""
        state.status = AgentStatus.PROCESSING
        return {"state": state, "next": "route"}

    def _route_message(self, state: AgentState) -> Dict[str, Any]:
        """Route message to appropriate handler"""
        message = state.memory.messages[-1]["content"].lower()

        # Check if this is about a previous paper
        if state.search_context.results and any(
            ref in message for ref in ["paper", "study", "research", "article"]
        ):
            return {"state": state, "next": "process_paper_question"}

        # Check if this is a search request
        if any(
            term in message for term in ["find", "search", "look for", "papers about"]
        ):
            return {"state": state, "next": "process_search"}

        # Default to conversation
        return {"state": state, "next": "process_conversation"}

    def _process_search(self, state: AgentState) -> Dict[str, Any]:
        """Process search request"""
        # Note: Actual search handling is done in EnhancedWorkflowManager
        return {"state": state, "next": "update_memory"}

    def _process_paper_question(self, state: AgentState) -> Dict[str, Any]:
        """Process paper-related questions"""
        # Note: Actual paper question handling is done in EnhancedWorkflowManager
        return {"state": state, "next": "update_memory"}

    def _process_conversation(self, state: AgentState) -> Dict[str, Any]:
        """Process general conversation"""
        # Note: Actual conversation handling is done in EnhancedWorkflowManager
        return {"state": state, "next": "update_memory"}

    def _update_memory(self, state: AgentState) -> Dict[str, Any]:
        """Update conversation memory"""
        if state.status != AgentStatus.ERROR:
            state.status = AgentStatus.SUCCESS
        return {"state": state, "next": END}

    def get_graph(self):
        """Get the compiled graph"""
        return self.graph.compile()

    def get_state(self) -> AgentState:
        """Get current state"""
        return self.current_state

    def reset_state(self):
        """Reset state to initial values"""
        self.current_state = AgentState()
