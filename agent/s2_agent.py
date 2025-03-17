import logging
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from config.config import config
from state.shared_state import Talk2Papers
from tools.s2 import s2_tools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class SemanticScholarAgent:
    def __init__(self):
        try:
            logger.info("Initializing S2 Agent...")
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            # Create single tools agent
            self.tools_agent = create_react_agent(
                self.llm,
                tools=s2_tools,
                state_schema=Talk2Papers,
                state_modifier=config.S2_AGENT_PROMPT,
                checkpointer=MemorySaver(),
            )

            def s2_supervisor_node(
                state: Talk2Papers,
            ) -> Command[Literal["tools_executor", "__end__"]]:
                """Internal supervisor for S2 agent"""
                logger.info("S2 supervisor node called")
                return Command(
                    goto="tools_executor",
                    update={
                        "current_agent": "s2_tools",
                        "messages": state.get("messages", []),
                        "is_last_step": False,
                    },
                )

            def tools_executor_node(state: Talk2Papers) -> Command[Literal["__end__"]]:
                """Execute tools based on request"""
                logger.info("Tools executor node called")
                try:
                    # Get tool execution result
                    result = self.tools_agent.invoke(state)
                    logger.info("Tool execution completed")

                    # Following documentation pattern exactly
                    return Command(
                        goto="__end__",
                        update={
                            "messages": [
                                HumanMessage(
                                    content=result["messages"][-1].content,
                                    name="s2_agent",
                                )
                            ],
                            "papers": result.get("papers", []),
                            "current_agent": None,
                            "is_last_step": True,
                        },
                    )
                except Exception as e:
                    logger.error(f"Error in tools executor: {str(e)}")
                    return Command(
                        goto="__end__",
                        update={
                            "messages": [
                                HumanMessage(
                                    content=f"Error: {str(e)}", name="s2_agent"
                                )
                            ],
                            "current_agent": None,
                            "is_last_step": True,
                        },
                    )

            # Create graph for S2 agent
            workflow = StateGraph(Talk2Papers)

            # Add nodes
            workflow.add_node("supervisor", s2_supervisor_node)
            workflow.add_node("tools_executor", tools_executor_node)

            # Add edges
            workflow.add_edge(START, "supervisor")

            # Compile with memory persistence
            self.graph = workflow.compile(checkpointer=MemorySaver())

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
            return {"error": str(e), "response": f"Error in S2 agent: {str(e)}"}


# Create a global instance
s2_agent = SemanticScholarAgent()
