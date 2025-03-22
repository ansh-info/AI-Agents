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

            # Create agent with tools and system prompt from config
            self.tools_agent = create_react_agent(
                self.llm,
                tools=s2_tools,
                state_schema=Talk2Papers,
                state_modifier=config.S2_AGENT_PROMPT,
            )

            def s2_supervisor_node(
                state: Talk2Papers,
            ) -> Command[Literal["tools_executor", "__end__"]]:
                """Internal supervisor for S2 agent"""
                logger.info("S2 supervisor node called")

                # Get decision from LLM
                try:
                    result = self.tools_agent.invoke(state)
                    if result.get("is_last_step", False):
                        return Command(
                            goto=END,
                            update={
                                "messages": result.get("messages", []),
                                "papers": result.get("papers", []),
                            },
                        )
                    return Command(goto="tools_executor", update=state)
                except Exception as e:
                    logger.error(f"Error in s2 supervisor: {str(e)}")
                    return Command(
                        goto=END,
                        update={"messages": [AIMessage(content=f"Error: {str(e)}")]},
                    )

            def tools_executor_node(state: Talk2Papers) -> Command[Literal["__end__"]]:
                """Execute tools based on request"""
                logger.info("Tools executor node called")
                try:
                    result = self.tools_agent.invoke(state)
                    logger.info("Tool execution completed")

                    return Command(
                        goto=END,
                        update={
                            "messages": result.get(
                                "messages", state.get("messages", [])
                            ),
                            "papers": result.get("papers", []),
                        },
                    )
                except Exception as e:
                    logger.error(f"Error in tools executor: {str(e)}")
                    return Command(
                        goto=END,
                        update={"messages": [AIMessage(content=f"Error: {str(e)}")]},
                    )

            # Create graph
            workflow = StateGraph(Talk2Papers)
            workflow.add_node("supervisor", s2_supervisor_node)
            workflow.add_node("tools_executor", tools_executor_node)

            workflow.add_edge(START, "supervisor")
            workflow.add_edge("tools_executor", END)

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
