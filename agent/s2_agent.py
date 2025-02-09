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
            self.llm = llm_manager.llm.bind_tools(
                [s2_tools[0]]
            )  # Only bind search_papers tool
            self.tool_executor = ToolExecutor(
                [s2_tools[0]]
            )  # Only use search_papers tool

            # Create prompt template with simplified structure
            self.prompt = ChatPromptTemplate.from_messages(
                [("system", config.S2_AGENT_PROMPT), ("human", "{input}")]
            )

            # Create the chain
            self.chain = self.prompt | self.llm

        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def parse_tool_call(self, content: str) -> Dict[str, Any]:
        """Parse tool call from JSON response"""
        try:
            tool_call = json.loads(content)
            if (
                isinstance(tool_call, dict)
                and "type" in tool_call
                and tool_call["type"] == "function"
            ):
                return {
                    "name": tool_call.get("name"),
                    "args": tool_call.get("parameters", {}),
                }
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
        return None

    def process_tool_calls(self, message: str, response: AIMessage) -> str:
        """Process tool calls and return final response"""
        try:
            print(f"Processing response content: {response.content}")
            tool_call = self.parse_tool_call(response.content)

            if tool_call:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                print(f"Executing tool: {tool_name} with args: {tool_args}")

                # Clean up tool arguments
                if "fields" in tool_args and isinstance(tool_args["fields"], str):
                    try:
                        tool_args["fields"] = json.loads(tool_args["fields"])
                    except json.JSONDecodeError:
                        tool_args["fields"] = None

                tool_output = self.tool_executor.invoke(tool_name, tool_args)
                print(f"Tool output: {tool_output}")

                if isinstance(tool_output, dict) and "papers" in tool_output:
                    papers = tool_output["papers"]
                    response_content = "Here are the papers I found:\n\n"
                    for i, paper in enumerate(papers, 1):
                        title = paper.get("title", "Untitled")
                        response_content += f"{i}. {title}\n"
                    return response_content
                else:
                    return f"Search completed but no papers were found. Tool output: {tool_output}"

            return response.content

        except Exception as e:
            print(f"Error in process_tool_calls: {e}")
            return f"Error processing search: {str(e)}"

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

                response_content = self.process_tool_calls(message, response)
                state["response"] = response_content

            except Exception as e:
                print(f"Error in message handling: {str(e)}")
                state["error"] = f"Error processing message: {str(e)}"
                return state

            print(f"Final response: {state.get('response')}")
            if state.get("response"):
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
