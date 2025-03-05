from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START
from state.shared_state import Talk2Papers
from tools.s2 import s2_tools
from langgraph.checkpoint.memory import MemorySaver


class SemanticScholarAgent:
    def __init__(self):
        try:
            print("Initializing S2 Agent...")

            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            self.agent = create_react_agent(
                self.llm,
                tools=s2_tools,
                state_schema=Talk2Papers,
                state_modifier=("You are Talk2Papers agent."),
                checkpointer=MemorySaver(),
            )

            def s2_node(state: Talk2Papers):
                result = self.agent.invoke(state)
                return Command(
                    update={
                        "messages": [
                            HumanMessage(
                                content=result["messages"][-1].content, name="s2_agent"
                            )
                        ]
                    },
                    goto="supervisor",
                )

            # Create graph for S2 agent
            workflow = StateGraph(Talk2Papers)
            workflow.add_node("s2_agent", s2_node)
            workflow.add_edge(START, "s2_agent")

            # Compile with memory persistence
            self.graph = workflow.compile(checkpointer=MemorySaver())

            print("S2 Agent initialized successfully")

        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def invoke(self, state):
        try:
            return self.graph.invoke(state)
        except Exception as e:
            return {"error": str(e), "response": f"Error in S2 agent: {str(e)}"}


# Create a global instance
s2_agent = SemanticScholarAgent()
