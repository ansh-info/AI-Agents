import logging
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from agents.s2_agent import s2_agent  # Import the S2 agent instance
from config.config import config
from state.shared_state import Talk2Papers
from tools.s2 import s2_tools  # Import tools for supervisor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def make_supervisor_node(llm: BaseChatModel) -> str:
    """
    Creates a supervisor node following LangGraph patterns.
    Uses the system prompt from config.
    """
    # Define available agents/options
    options = ["FINISH", "s2_agent", "zotero_agent", "pdf_agent", "arxiv_agent"]

    # Get system prompt from config
    system_prompt = config.MAIN_AGENT_PROMPT

    def supervisor_node(
        state: Talk2Papers,
    ) -> Command[
        Literal["s2_agent", "zotero_agent", "pdf_agent", "arxiv_agent", "__end__"]
    ]:
        """Supervisor node that routes to appropriate sub-agents"""
        logger.info("Supervisor node called")
        logger.info(f"Current state: {state.get('current_agent')}")

        # Create messages list with system prompt
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]

        # Create agent with main supervisor prompt
        supervisor_agent = create_react_agent(
            llm,
            tools=s2_tools,
            state_schema=Talk2Papers,
            state_modifier=system_prompt,
            checkpointer=MemorySaver(),
        )

        # Get routing decision
        result = supervisor_agent.invoke(state)
        logger.info(f"Supervisor decision: {result}")

        # For now, we only have s2_agent implemented
        # In future, this will be determined by the agent's response
        goto = "s2_agent"
        if goto == "FINISH":
            goto = END

        return Command(
            goto=goto,
            update={
                "next": goto,
                "messages": result.get("messages", state.get("messages", [])),
                "current_agent": goto if goto != END else None,
                "is_last_step": goto == END,
            },
        )

    return supervisor_node


def call_s2_agent(state: Talk2Papers) -> Command[Literal["supervisor", "__end__"]]:
    """Node for calling the S2 agent"""
    logger.info("Calling S2 agent")
    try:
        # Call the S2 agent
        response = s2_agent.invoke(state)
        logger.info("S2 agent completed")

        # Following LangGraph patterns
        return Command(
            goto="supervisor",
            update={
                "messages": [
                    HumanMessage(
                        content=response["messages"][-1].content, name="s2_agent"
                    )
                ],
                "papers": response.get("papers", []),
                "is_last_step": False,
            },
        )
    except Exception as e:
        logger.error(f"Error in S2 agent node: {str(e)}")
        return Command(
            goto="supervisor",
            update={
                "messages": [HumanMessage(content=f"Error: {str(e)}", name="s2_agent")],
                "is_last_step": False,
            },
        )


def get_app(thread_id: str):
    """
    Returns the langraph app with hierarchical structure.
    """
    # Create the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create the supervisory graph
    workflow = StateGraph(Talk2Papers)

    # Create supervisor node
    supervisor = make_supervisor_node(llm)

    # Add nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("s2_agent", call_s2_agent)

    # Add edges following LangGraph patterns
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("s2_agent", "supervisor")

    # Initialize memory
    checkpointer = MemorySaver()

    # Compile the workflow
    app = workflow.compile(checkpointer=checkpointer)

    logger.info("Main agent workflow compiled")
    return app
