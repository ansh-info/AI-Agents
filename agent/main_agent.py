import logging
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from config.config import config
from state.shared_state import Talk2Papers

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def make_supervisor_node(llm: BaseChatModel) -> str:
    """Creates a supervisor node following LangGraph patterns."""
    options = ["FINISH", "s2_agent"]  # Only implemented agents
    system_prompt = config.MAIN_AGENT_PROMPT

    def supervisor_node(state: Talk2Papers) -> Command[Literal["s2_agent", "__end__"]]:
        """Supervisor node that routes to appropriate sub-agents"""
        logger.info("Supervisor node called")
        logger.info(f"Current state: {state.get('current_agent')}")

        # Create messages list with system prompt
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]

        # Get response from LLM
        response = llm.invoke(messages)

        # Check if we should finish or continue
        goto = "FINISH" if "search" not in response.content.lower() else "s2_agent"
        if goto == "FINISH":
            return Command(
                goto=END, update={"is_last_step": True, "messages": state["messages"]}
            )

        return Command(goto=goto, update={"next": goto})

    return supervisor_node


def call_s2_agent(state: Talk2Papers) -> Command[Literal["supervisor"]]:
    """Node for calling the S2 agent"""
    logger.info("Calling S2 agent")
    try:
        response = s2_agent.invoke(state)
        logger.info("S2 agent completed")

        return Command(
            goto="supervisor",  # Always return to supervisor for next decision
            update={
                "messages": [HumanMessage(content=response["messages"][-1].content)],
                "papers": response.get("papers", []),
            },
        )
    except Exception as e:
        logger.error(f"Error in S2 agent node: {str(e)}")
        return Command(
            goto="supervisor",
            update={"messages": [HumanMessage(content=f"Error: {str(e)}")]},
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
