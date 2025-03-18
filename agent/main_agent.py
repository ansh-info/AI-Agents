import logging
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from config.config import config
from state.shared_state import Talk2Papers
from agents.s2_agent import s2_agent  # Import the S2 agent instance
from tools.s2 import s2_tools  # Import tools for supervisor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def get_app(thread_id: str):
    """
    This function returns the langraph app with hierarchical structure.

    Args:
        thread_id (str): Thread ID for the MemorySaver checkpointer
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
            tools=s2_tools,  # Following documentation - supervisor needs tools
            state_schema=Talk2Papers,
            state_modifier=config.MAIN_AGENT_PROMPT,
            checkpointer=MemorySaver(),
        )

        # Get routing decision
        result = supervisor_agent.invoke(state)
        logger.info(f"Supervisor decision: {result}")

        # For now, we only have s2_agent
        goto = "s2_agent"
        return Command(
            goto=goto,
            update={
                "next": goto,
                "messages": result.get("messages", state.get("messages", [])),
                "is_last_step": False,
            },
        )

    def call_s2_agent(state: Talk2Papers) -> Command[Literal["supervisor", "__end__"]]:
        """Node for calling the S2 agent"""
        logger.info("Calling S2 agent")
        try:
            # Call the S2 agent
            response = s2_agent.invoke(state)
            logger.info("S2 agent completed")

            # Following documentation pattern
            return Command(
                goto=END,
                update={
                    "next": END,
                    "messages": response.get("messages", []),
                    "papers": response.get("papers", []),
                    "is_last_step": True,
                },
            )
        except Exception as e:
            logger.error(f"Error in S2 agent node: {str(e)}")
            return Command(
                goto=END,
                update={
                    "next": END,
                    "messages": [{"role": "assistant", "content": f"Error: {str(e)}"}],
                    "is_last_step": True,
                },
            )

    # Create the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Define the supervisory graph
    workflow = StateGraph(Talk2Papers)

    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("s2_agent", call_s2_agent)

    # Add edges
    workflow.add_edge(START, "supervisor")

    # Initialize memory
    checkpointer = MemorySaver()

    # Compile the workflow
    app = workflow.compile(checkpointer=checkpointer)

    logger.info("Main agent workflow compiled")
    return app
