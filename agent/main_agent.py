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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def make_supervisor_node(llm: BaseChatModel) -> str:
    """Creates a supervisor node following LangGraph patterns."""

    def supervisor_node(state: Talk2Papers) -> Command[Literal["s2_agent", "__end__"]]:
        """Supervisor node that routes to appropriate sub-agents"""
        logger.info("Supervisor node called")
        logger.info(f"Current state: {state.get('current_agent')}")

        # Create messages list with config system prompt
        messages = [{"role": "system", "content": config.MAIN_AGENT_PROMPT}] + state[
            "messages"
        ]

        # Create supervisor agent with tools
        supervisor_agent = create_react_agent(
            llm, tools=s2_tools, system_message=config.MAIN_AGENT_PROMPT
        )

        # Get routing decision and response
        result = supervisor_agent.invoke(state)
        response = result["messages"][-1].content
        logger.info(f"LLM response: {response}")

        # Check if this is a paper-related query
        paper_related = any(
            term in state["messages"][-1].content.lower()
            for term in ["paper", "research", "academic", "find", "search"]
        )

        if paper_related:
            # Route to S2 agent
            return Command(
                goto="s2_agent",
                update={
                    "messages": state["messages"],
                    "current_agent": "s2_agent",
                    "is_last_step": False,
                    "papers": [],
                },
            )
        else:
            # Handle general conversation
            return Command(
                goto=END,
                update={
                    "messages": state["messages"] + [AIMessage(content=response)],
                    "papers": [],
                    "current_agent": None,
                    "is_last_step": True,
                },
            )

    return supervisor_node


def call_s2_agent(state: Talk2Papers) -> Command[Literal["__end__"]]:
    """Node for calling the S2 agent"""
    logger.info("Calling S2 agent")
    try:
        result = s2_agent.invoke(state)
        logger.info("S2 agent completed")

        return Command(
            goto=END,
            update={
                "messages": result["messages"],
                "papers": result.get("papers", []),
                "current_agent": "s2_agent",
                "is_last_step": True,
                "search_table": result.get("search_table", ""),
            },
        )
    except Exception as e:
        logger.error(f"Error in S2 agent node: {str(e)}")
        return Command(
            goto=END,
            update={
                "messages": state["messages"] + [AIMessage(content=f"Error: {str(e)}")],
                "current_agent": "s2_agent",
                "is_last_step": True,
            },
        )


def get_app(thread_id: str):
    """Returns the langraph app with hierarchical structure."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    workflow = StateGraph(Talk2Papers)

    supervisor = make_supervisor_node(llm)
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("s2_agent", call_s2_agent)

    workflow.add_edge(START, "supervisor")
    workflow.add_edge("s2_agent", END)

    app = workflow.compile(checkpointer=MemorySaver())
    logger.info("Main agent workflow compiled")
    return app
