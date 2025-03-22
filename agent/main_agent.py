import logging
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from config.config import config
from state.shared_state import Talk2Papers
from agents.s2_agent import s2_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def make_supervisor_node(llm: BaseChatModel) -> str:
    """Creates a supervisor node following LangGraph patterns."""
    # Define available options
    options = ["FINISH", "s2_agent"]

    def supervisor_node(state: Talk2Papers) -> Command[Literal["s2_agent", "__end__"]]:
        """Supervisor node that routes to appropriate sub-agents"""
        logger.info("Supervisor node called")

        # Create messages list with system prompt from config
        messages = [{"role": "system", "content": config.MAIN_AGENT_PROMPT}] + state[
            "messages"
        ]

        # Get routing decision from LLM
        response = llm.invoke(messages)
        logger.info(f"LLM routing response: {response.content}")

        # Parse decision
        goto = response.content.strip().lower()

        # Convert FINISH to END
        if goto == "finish":
            return Command(
                goto=END, update={"messages": state["messages"], "is_last_step": True}
            )

        # Route to S2 agent
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
        # Call the S2 agent
        response = s2_agent.invoke(state)
        logger.info("S2 agent completed")

        # Return to supervisor for next decision
        return Command(
            goto="supervisor",
            update={
                "messages": state["messages"]
                + [AIMessage(content=response["messages"][-1].content)],
                "papers": response.get("papers", []),
                "current_agent": "s2_agent",
            },
        )
    except Exception as e:
        logger.error(f"Error in S2 agent node: {str(e)}")
        return Command(
            goto="supervisor",
            update={
                "messages": state["messages"] + [AIMessage(content=f"Error: {str(e)}")]
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
    workflow.add_edge("s2_agent", "supervisor")

    app = workflow.compile(checkpointer=MemorySaver())
    logger.info("Main agent workflow compiled")
    return app
