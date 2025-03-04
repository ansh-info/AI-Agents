from typing import Any, Dict
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from config.config import config
from state.shared_state import shared_state
from agents.s2_agent import s2_agent
from utils.llm import llm_manager


class MainAgent:
    def __init__(self):
        try:
            # Define routing tools using StructuredTool instead of BaseTool
            self.routing_tools = [
                StructuredTool.from_function(
                    func=self.route_to_s2_agent,
                    name="semantic_scholar_agent",
                    description="""Use for any academic paper search, finding papers, or getting paper recommendations. 
                    Best for: finding research papers, academic search, paper recommendations""",
                ),
                # Add other agent tools as they are implemented
                # StructuredTool.from_function(name="zotero_agent", ...),
                # StructuredTool.from_function(name="pdf_agent", ...),
                # StructuredTool.from_function(name="arxiv_agent", ...),
            ]

            # Create the agent using create_react_agent
            self.agent = create_react_agent(
                model=llm_manager.llm,
                tools=self.routing_tools,
                messages_modifier=config.MAIN_AGENT_PROMPT,
            )

        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def route_to_s2_agent(self, query: str) -> Dict[str, Any]:
        """Route queries to Semantic Scholar agent.

        Args:
            query: The user's query about academic papers or research.

        Returns:
            Dict containing the response from the S2 agent.
        """
        try:
            shared_state.set(config.StateKeys.CURRENT_AGENT, config.AgentNames.S2)
            result = s2_agent.invoke({"messages": [{"role": "user", "content": query}]})
            return {
                "status": "success",
                "response": (
                    result["messages"][-1].content
                    if result.get("messages")
                    else "No response from S2 agent"
                ),
            }
        except Exception as e:
            return {"status": "error", "response": f"Error in S2 agent: {str(e)}"}

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input state through the agent"""
        try:
            return self.agent.invoke(state)
        except Exception as e:
            return {"error": str(e), "response": f"Error in main agent: {str(e)}"}


# Create a global instance
main_agent = MainAgent()
