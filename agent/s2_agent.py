from typing import Any, Dict
from langgraph.prebuilt import create_react_agent
from config.config import config
from state.shared_state import shared_state
from tools.s2 import s2_tools
from utils.llm import llm_manager


class SemanticScholarAgent:
    def __init__(self):
        try:
            print("Initializing S2 Agent...")

            # Create the agent using create_react_agent
            self.agent = create_react_agent(
                model=llm_manager.llm,
                tools=s2_tools,
                messages_modifier=config.S2_AGENT_PROMPT,  # Changed from system_message
            )

            print("S2 Agent initialized successfully")

        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input state through the agent"""
        try:
            shared_state.set(config.StateKeys.CURRENT_AGENT, config.AgentNames.S2)
            return self.agent.invoke(state)
        except Exception as e:
            return {"error": str(e), "response": f"Error in S2 agent: {str(e)}"}


# Create a global instance
s2_agent = SemanticScholarAgent()
