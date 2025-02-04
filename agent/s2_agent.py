from typing import Any, Dict

from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from config.config import config
from state.shared_state import shared_state
from tools.s2.search import s2_search_tool
from utils.llm import llm_manager


class SemanticScholarAgent:
    def __init__(self):
        self.llm = llm_manager
        self.llm.bind_tools(
            [
                self.search_papers,
                # We'll add other tools later:
                # self.get_single_paper_recommendations,
                # self.get_multi_paper_recommendations
            ]
        )

    @tool
    def search_papers(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search for academic papers on Semantic Scholar.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            Dict containing search results
        """
        return s2_search_tool.search_papers(query, limit=limit)

    def handle_message(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming messages and route to appropriate tool"""
        try:
            # Get current message and context
            message = state.get("message", "")
            context = shared_state.get_current_context()

            # Get LLM response with tool execution
            response = self.llm.get_response(
                system_prompt=config.S2_AGENT_PROMPT,
                user_input=message,
                additional_context=context,
            )

            # Update state and shared state
            state["response"] = response
            shared_state.add_to_chat_history("assistant", response)

            return state

        except Exception as e:
            error_msg = f"Error in S2 agent: {str(e)}"
            state["error"] = error_msg
            shared_state.set(config.StateKeys.ERROR, error_msg)
            return state

    def create_graph(self) -> StateGraph:
        """Create the agent's workflow graph"""
        workflow = StateGraph()

        # Add nodes
        workflow.add_node("process_message", self.handle_message)

        # Add edges
        workflow.add_edge("process_message", END)

        # Set entry point
        workflow.set_entry_point("process_message")

        return workflow


# Create a global instance
s2_agent = SemanticScholarAgent()
