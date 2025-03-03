from typing import Any, Dict, Literal
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool
from config.config import config
from state.shared_state import shared_state
from utils.llm import llm_manager
from agents.s2_agent import s2_agent


class MainAgent:
    def __init__(self):
        self.llm = llm_manager.llm
        # Define our routing tools for the main supervisor
        self.routing_tools = [
            BaseTool(
                name="semantic_scholar_agent",
                description="Routes queries about academic papers, paper search, and recommendations to Semantic Scholar agent",
                func=self.route_to_s2_agent,
                args_schema=None,
            ),
            # Add other agents as tools here when implemented
            # BaseTool(name="zotero_agent",...),
            # BaseTool(name="pdf_agent",...),
            # BaseTool(name="arxiv_agent",...)
        ]

        # Create the supervisor agent using create_react_agent
        self.agent = create_react_agent(
            self.llm, tools=self.routing_tools, system_message=config.MAIN_AGENT_PROMPT
        )

    def route_to_s2_agent(self, query: str) -> Dict[str, Any]:
        """Route to Semantic Scholar agent"""
        try:
            shared_state.set(config.StateKeys.CURRENT_AGENT, config.AgentNames.S2)
            result = s2_agent.invoke({"messages": [HumanMessage(content=query)]})
            return {
                "response": (
                    result["messages"][-1].content
                    if result.get("messages")
                    else "No response from S2 agent"
                ),
                "status": "success",
            }
        except Exception as e:
            return {"response": f"Error in S2 agent: {str(e)}", "status": "error"}

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input state through the agent"""
        try:
            return self.agent.invoke(state)
        except Exception as e:
            return {"error": str(e), "response": f"Error in main agent: {str(e)}"}


# Create a global instance
main_agent = MainAgent()
