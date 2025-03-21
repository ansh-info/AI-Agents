import logging
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from config.config import config
from state.shared_state import Talk2Papers
from agents.s2_agent import s2_agent
from tools.s2 import s2_tools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def make_supervisor_node(llm: BaseChatModel) -> str:
    """Creates a supervisor node following LangGraph patterns."""
    options = ["FINISH", "s2_agent"]  # Only implemented agents
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the "
        f"following agents: {options}. Given the following user request, "
        "respond with the agent to act next or FINISH if the request can be "
        "handled directly. For paper-related queries, always use s2_agent."
    )

    def supervisor_node(state: Talk2Papers) -> Command[Literal["s2_agent", "__end__"]]:
        """Supervisor node that routes to appropriate sub-agents"""
        logger.info("Supervisor node called")
        logger.info(f"Current state: {state.get('current_agent')}")

        # Create messages list with system prompt
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]

        # Get routing decision from LLM
        response = llm.invoke(messages)
        response_text = response.content.lower()
        logger.info(f"LLM routing response: {response_text}")

        # Check if this is a paper-related query
        paper_related = any(
            term in response_text
            for term in ["paper", "research", "academic", "find", "search"]
        )

        if not paper_related:
            # Handle general conversation
            return Command(
                goto=END,
                update={
                    "messages": state["messages"]
                    + [AIMessage(content=response.content)],
                    "papers": [],
                    "current_agent": None,
                    "is_last_step": True,
                },
            )

        # Route to S2 agent for paper-related queries
        return Command(
            goto="s2_agent",
            update={
                "messages": state["messages"],
                "current_agent": "s2_agent",
                "is_last_step": False,
            },
        )

    return supervisor_node


def call_s2_agent(state: Talk2Papers) -> Command[Literal["supervisor"]]:
    """Node for calling the S2 agent"""
    logger.info("Calling S2 agent")
    try:
        response = s2_agent.invoke(state)
        logger.info("S2 agent completed")

        # Always return to supervisor for next decision
        return Command(
            goto="supervisor",
            update={
                "messages": state["messages"]
                + [AIMessage(content=response["messages"][-1].content)],
                "papers": response.get("papers", []),
                "current_agent": "s2_agent",
                "is_last_step": False,
            },
        )
    except Exception as e:
        logger.error(f"Error in S2 agent node: {str(e)}")
        return Command(
            goto="supervisor",
            update={
                "messages": state["messages"] + [AIMessage(content=f"Error: {str(e)}")],
                "current_agent": None,
                "is_last_step": True,
            },
        )


def get_app(thread_id: str):
    """Returns the langraph app with hierarchical structure."""
    # Create the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create the graph
    workflow = StateGraph(Talk2Papers)

    # Create supervisor node
    supervisor = make_supervisor_node(llm)

    # Add nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("s2_agent", call_s2_agent)

    # Add edges
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("s2_agent", "supervisor")

    # Compile with memory
    app = workflow.compile(checkpointer=MemorySaver())

    logger.info("Main agent workflow compiled")
    return app
