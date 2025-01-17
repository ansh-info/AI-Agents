from typing import Any, Dict, Optional

from agent.main_agent import MainAgent
from clients.semantic_scholar_client import SemanticScholarClient
from state.agent_state import AgentState, AgentStatus
from tools.research_tools import ResearchTools


class ResearchWorkflowManager:
    """Manages the research workflow and agent interactions."""

    def __init__(self, model_name: str = "llama3.2:1b-instruct-q3_K_M"):
        # Initialize clients
        self.s2_client = SemanticScholarClient()

        # Initialize tools
        self.research_tools = ResearchTools()

        # Create tools list
        self.tools = [
            self.research_tools.search_papers,
            self.research_tools.get_paper_details,
        ]

        # Initialize main agent
        self.main_agent = MainAgent(tools=self.tools, model_name=model_name)

        # Initialize state
        self.current_state = AgentState()

    async def process_message(self, message: str) -> AgentState:
        """Process a user message through the workflow."""
        try:
            print(f"[DEBUG] WorkflowManager processing message: {message}")

            # Update state with new message
            self.current_state.add_message("user", message)

            # Process through main agent
            state = await self.main_agent.process_request(self.current_state)

            # If no response was generated, add a default response
            if len(state.memory.messages) == len(self.current_state.memory.messages):
                state.add_message(
                    "system",
                    "I understand you're trying to search for research papers. Could you please specify what topic you'd like to search for?",
                )

            return state

        except Exception as e:
            print(f"[DEBUG] Error in workflow: {str(e)}")
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            self.current_state.add_message(
                "system", f"I apologize, but I encountered an error: {str(e)}"
            )
            return self.current_state

    async def check_health(self) -> Dict[str, bool]:
        """Check the health of all components."""
        try:
            return {
                "semantic_scholar": await self.s2_client.check_api_status(),
                "main_agent": True,  # Add more specific checks if needed
                "tools": all(
                    hasattr(self.research_tools, tool.__name__) for tool in self.tools
                ),
            }
        except Exception as e:
            return {
                "semantic_scholar": False,
                "main_agent": False,
                "tools": False,
                "error": str(e),
            }
