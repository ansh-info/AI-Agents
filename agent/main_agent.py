"""
This is the agent file for the Talk2Papers graph.
"""

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from typing import Literal

from state.shared_state import Talk2Papers
from tools.s2 import s2_tools
from agents.s2_agent import s2_agent
from config.config import config


def get_app(uniq_id):
    """
    This function returns the langraph app.
    """

    def agent_test_node(state: Talk2Papers):
        """
        This function calls the model.
        """
        # Get the messages from the state
        messages = state["messages"]
        # Call the model
        inputs = {"messages": messages}
        response = model.invoke(inputs, {"configurable": {"thread_id": uniq_id}})
        # Return response
        return response

    # Define the tools
    tools = [s2_tools[0], s2_tools[1]]

    # Create the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    model = create_react_agent(
        llm,
        tools=tools,
        state_schema=Talk2Papers,
        state_modifier=config.MAIN_AGENT_PROMPT,
        checkpointer=MemorySaver(),
    )

    # Define a new graph
    workflow = StateGraph(Talk2Papers)

    # Define the node
    workflow.add_node("agent_test", agent_test_node)

    # Set the entrypoint
    workflow.add_edge(START, "agent_test")

    # Initialize memory
    checkpointer = MemorySaver()

    # Compile the workflow
    app = workflow.compile(checkpointer=checkpointer)

    return app
