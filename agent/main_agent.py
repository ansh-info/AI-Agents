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
from agents.s2_agent import s2_agent  # Import the S2 agent instance

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
        Supervisor node that routes to appropriate sub-agents based on the main agent prompt.
        Currently only routes to s2_agent as other agents are not implemented.
        """
        logger.info("Supervisor node called")
        logger.info(f"Current state: {state.get('current_agent')}")

        # Create agent with main supervisor prompt
        supervisor_agent = create_react_agent(
            llm,
            state_schema=Talk2Papers,
            state_modifier=config.MAIN_AGENT_PROMPT,
            checkpointer=MemorySaver(),
        )

        # Get routing decision
        result = supervisor_agent.invoke(state)
        logger.info(f"Supervisor decision: {result}")

        # For now, we only have s2_agent
        return Command(
            goto="s2_agent",
            update={
                "current_agent": "s2_agent",
                "next": "s2_agent",
                "messages": result.get("messages", state.get("messages", [])),
                "is_last_step": False,
            },
        )

    def s2_agent_node(state: Talk2Papers) -> Command[Literal["supervisor", "__end__"]]:
        """Node for calling the S2 agent"""
        logger.info("Calling S2 agent")
        try:
            # Call the S2 agent
            result = s2_agent.invoke(state)
            logger.info("S2 agent completed")

            # Return to supervisor if more work needed, otherwise end
            return Command(
                goto="__end__",
                update={
                    "messages": result.get("messages", []),
                    "papers": result.get("papers", []),
                    "current_agent": None,
                },
            )
        except Exception as e:
            logger.error(f"Error in S2 agent node: {str(e)}")
            return Command(
                goto="__end__",
                update={
                    "messages": [{"role": "assistant", "content": f"Error: {str(e)}"}],
                    "current_agent": None,
                },
            )

    # Create the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Define the supervisory graph
    workflow = StateGraph(Talk2Papers)

    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("s2_agent", s2_agent_node)

    # Add edges
    workflow.add_edge(START, "supervisor")

    # Initialize memory
    checkpointer = MemorySaver()

    # Compile the workflow
    app = workflow.compile(checkpointer=checkpointer)

    logger.info("Main agent workflow compiled")
    return app
