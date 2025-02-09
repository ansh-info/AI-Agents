from typing import List, Dict, Any, Type, TypedDict
import json
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor

from config.config import config
from state.shared_state import shared_state
from tools.s2.search import s2_tools
from utils.llm import llm_manager


class S2AgentState(TypedDict):
    """Type definition for S2 agent state"""
    message: str
    response: str | None
    error: str | None


class SemanticScholarAgent:
    def __init__(self):
        try:
            print("Initializing S2 Agent...")
            self.llm = llm_manager.llm.bind_tools([s2_tools[0]])  # Only bind search_papers tool
            self.tool_executor = ToolExecutor([s2_tools[0]])  # Only use search_papers tool

            # Create prompt template with simplified structure
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", config.S2_AGENT_PROMPT),
                ("human", "{input}")
            ])

            # Create the chain
            self.chain = self.prompt | self.llm

        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def handle_message(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming messages and route to appropriate tool"""
        try:
            print("\nHandling message...")
            message = state.get("message", "")
            print(f"Received message: {message}")

            try:
                print("Getting LLM response...")
                response = self.chain.invoke({"input": message})
                print(f"LLM Response type: {type(response)}")
                print(f"LLM Response content: {response.content}")
                
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    print(f"Tool calls found: {response.tool_calls}")
                    for tool_call in response.tool_calls:
                        try:
                            tool_name = tool_call.name
                            tool_args = tool_call.args
                            tool_id = tool_call.id

                            print(f"Executing tool: {tool_name} with args: {tool_args}")
                            tool_output = self.tool_executor.invoke(
                                tool_name, tool_args
                            )
                            print(f"Tool output: {tool_output}")

                            response = self.llm.invoke([
                                HumanMessage(content=message),
                                AIMessage(content=response.content, tool_calls=[tool_call]),
                                ToolMessage(content=str(tool_output), tool_call_id=tool_id)
                            ])
                    state["response"] = response.content
                else:
                    print("No tool calls found, using direct response")
                    state["response"] = response.content

            except Exception as e:
                print(f"Error in message handling: {str(e)}")
                state["error"] = f"Error processing message: {str(e)}"
                return state

            print(f"Final response: {state.get('response')}")
            if state.get('response'):
                shared_state.add_to_chat_history("assistant", state["response"])
            return state

        except Exception as e:
            print(f"Error in handle_message: {str(e)}")
            state["error"] = f"Error in S2 agent: {str(e)}"
            shared_state.set(config.StateKeys.ERROR, state["error"])
            return state

    def create_graph(self) -> StateGraph:
        """Create the agent's workflow graph"""
        workflow = StateGraph(S2AgentState)
        workflow.add_node("process_message", self.handle_message)
        workflow.add_edge("process_message", END)
        workflow.set_entry_point("process_message")
        return workflow.compile()


# Create a global instance
s2_agent = SemanticScholarAgent()
