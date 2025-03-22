import logging
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from config.config import config
from state.shared_state import Talk2Papers
from tools.s2 import s2_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class SemanticScholarAgent:
    def __init__(self):
        try:
            logger.info("Initializing S2 Agent...")
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            # Create the tools agent using config prompt
            self.tools_agent = create_react_agent(
                self.llm,
                tools=s2_tools,
                state_schema=Talk2Papers,
                state_modifier=config.S2_AGENT_PROMPT,
            )

            def execute_tools(state: Talk2Papers) -> Command[Literal["__end__"]]:
                """Execute tools and return results"""
                logger.info("Executing tools")
                try:
                    result = self.tools_agent.invoke(state)
                    logger.info("Tool execution completed")

                    return Command(
                        goto=END,
                        update={
                            "messages": result["messages"],
                            "papers": result.get("papers", []),
                            "is_last_step": True,
                        },
                    )
                except Exception as e:
                    logger.error(f"Error executing tools: {str(e)}")
                    return Command(
                        goto=END,
                        update={
                            "messages": [AIMessage(content=f"Error: {str(e)}")],
                            "is_last_step": True,
                        },
                    )

            # Create graph
            workflow = StateGraph(Talk2Papers)
            workflow.add_node("tools", execute_tools)
            workflow.add_edge(START, "tools")

            self.graph = workflow.compile()
            logger.info("S2 Agent initialized successfully")

        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise

    def invoke(self, state):
        try:
            logger.info("Invoking S2 agent")
            return self.graph.invoke(state)
        except Exception as e:
            logger.error(f"Error in S2 agent: {str(e)}")
            return {"messages": [AIMessage(content=f"Error: {str(e)}")], "papers": []}


# Create a global instance
s2_agent = SemanticScholarAgent()
