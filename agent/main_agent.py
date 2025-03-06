"""
This is the agent file for the Talk2Papers graph.
"""

from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langchain.prebuilt.llm_functions import (
    make_supervisor_node,
)

from state.shared_state import Talk2Papers
from tools.s2 import s2_tools
from config.config import config
from agents.s2_agent import s2_agent


def get_app(uniq_id):  # Change parameter name to match what we use inside
    """
    This function returns the langraph app.
    """

    def call_s2_agent(state: Talk2Papers) -> Command[Literal["supervisor"]]:
        response = s2_agent.invoke({"messages": state["messages"][-1]})
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=response["messages"][-1].content, name="s2_agent"
                    )
                ]
            },
            goto="supervisor",
        )

    # Create the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    model = create_react_agent(
        llm,
        tools=tools,
        state_schema=Talk2Papers,
        state_modifier=config.MAIN_AGENT_PROMPT,
        checkpointer=MemorySaver(),
    )

    # Create supervisor node
    supervisor_node = make_supervisor_node(llm, ["s2_agent"])

    # Define the graph
    workflow = StateGraph(Talk2Papers)

    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("s2_agent", call_s2_agent)

    # Set the entrypoint
    workflow.add_edge(START, "supervisor")

    # Initialize memory
    checkpointer = MemorySaver()

    # Compile the workflow
    app = workflow.compile(checkpointer=checkpointer)

    return app
