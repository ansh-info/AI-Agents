import logging
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from config.config import config
from state.shared_state import Talk2Papers

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def get_app(uniq_id):
    """
    This function returns the langraph app with hierarchical structure.
    """

    def supervisor_node(state: Talk2Papers) -> Command[Literal["s2_agent", "__end__"]]:
        """
        Supervisor node that routes to appropriate sub-agents.
        Currently only routes to s2_agent as other agents are not implemented.
        """
        logger.info("Supervisor node called")
        logger.info(f"Current state: {state.get('current_agent')}")

        # For now, we only have s2_agent
        return Command(
            goto="s2_agent", update={"current_agent": "s2_agent", "next": "s2_agent"}
        )

    # Create the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Define the supervisory graph
    workflow = StateGraph(Talk2Papers)

    # Add nodes
    workflow.add_node("supervisor", supervisor_node)

    # Add edges
    workflow.add_edge(START, "supervisor")

    # Initialize memory
    checkpointer = MemorySaver()

    # Compile the workflow
    app = workflow.compile(checkpointer=checkpointer)

    logger.info("Main agent workflow compiled")
    return app
