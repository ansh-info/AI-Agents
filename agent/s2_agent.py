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
            self.llm = llm_manager.llm.bind_tools(s2_tools)
            self.tool_executor = ToolExecutor(s2_tools)

            # Create prompt template
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", config.S2_AGENT_PROMPT),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}"),
                ]
            )

            # Create the chain
            self.chain = (
                RunnablePassthrough.assign(history=lambda x: [])
                | self.prompt
                | self.llm
            )
            print("S2 Agent initialized successfully")
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def handle_message(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming messages and route to appropriate tool"""
        try:
            print("\nHandling message...")
            message = state.get("message", "")
            print(f"Received message: {message}")

            context = shared_state.get_current_context()
            if context:
                message += f"\n\nContext:\n{context}"
                print("Added context to message")

            try:
                print("Getting LLM response...")
                response = self.chain.invoke({"input": message})
                print(f"LLM Response received: {str(response)}")
                print(f"Response type: {type(response)}")
                print(f"Response attributes: {dir(response)}")
                if hasattr(response, "additional_kwargs"):
                    print(f"Additional kwargs: {response.additional_kwargs}")
            except ConnectionError as conn_err:
                state["error"] = (
                    f"Connection error: Please ensure Ollama service is running. Error: {str(conn_err)}"
                )
                return state
            except Exception as chain_err:
                state["error"] = f"Chain error: {str(chain_err)}"
                return state

            messages = [HumanMessage(content=message), response]

            if (
                hasattr(response, "additional_kwargs")
                and "tool_calls" in response.additional_kwargs
            ):
                print("Tool calls found in response")
                tool_calls = response.additional_kwargs["tool_calls"]
                print(f"Tool calls: {json.dumps(tool_calls, indent=2)}")

                for tool_call in tool_calls:
                    try:
                        tool_name = tool_call.get("function", {}).get("name")
                        tool_args = tool_call.get("function", {}).get("arguments", {})
                        tool_id = tool_call.get("id", "default_id")

                        print(f"Executing tool: {tool_name}")
                        print(f"Tool arguments: {tool_args}")

                        if tool_name and tool_args:
                            if isinstance(tool_args, str):
                                try:
                                    tool_args = json.loads(tool_args)
                                except json.JSONDecodeError:
                                    print(
                                        f"Could not parse tool args as JSON: {tool_args}"
                                    )

                            tool_output = self.tool_executor.invoke(
                                tool_name, tool_args
                            )
                            print(f"Tool execution output: {tool_output}")

                            messages.append(
                                ToolMessage(
                                    content=str(tool_output), tool_call_id=tool_id
                                )
                            )
                    except Exception as tool_error:
                        print(f"Tool execution error: {str(tool_error)}")
                        messages.append(
                            ToolMessage(
                                content=str(
                                    {"error": f"Tool error: {str(tool_error)}"}
                                ),
                                tool_call_id=(
                                    tool_id if "tool_id" in locals() else "error_id"
                                ),
                            )
                        )

                try:
                    print("Getting final response from LLM...")
                    final_response = self.llm.invoke(messages)
                    response_content = final_response.content
                    print(f"Final response: {response_content}")
                except Exception as llm_err:
                    print(f"Error getting final response: {str(llm_err)}")
                    state["error"] = f"LLM response error: {str(llm_err)}"
                    return state
            else:
                print("No tool calls found in response")
                response_content = response.content

            print(f"Setting final response: {response_content}")
            state["response"] = response_content
            shared_state.add_to_chat_history("assistant", response_content)
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
