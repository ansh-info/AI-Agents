import json
import traceback
from typing import Any, Dict, List, Type, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor

from config.config import config
from state.shared_state import shared_state
from tools.s2.search import search_papers
from tools.s2.single_paper_rec import get_single_paper_recommendations
from tools.s2.multi_paper_rec import get_multi_paper_recommendations
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

            # Store the tools
            self.tools = {
                "search_papers": search_papers,
                "get_single_paper_recommendations": get_single_paper_recommendations,
                "get_multi_paper_recommendations": get_multi_paper_recommendations,
            }

            # Configure LLM with lower temperature for more consistent responses
            self.llm = llm_manager.llm.bind(
                options={
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "repeat_penalty": 1.1,
                    "num_ctx": 2048,
                }
            )

            # Create prompt template
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", config.S2_AGENT_PROMPT),
                    MessagesPlaceholder("chat_history"),  # Add chat history support
                    ("human", "{input}"),
                ]
            )

            # Create chain with response validation
            self.chain = self.prompt | self.llm

            print("S2 Agent initialized successfully")

        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the selected tool"""
        try:
            tool_name = tool_call.get("name")
            raw_args = tool_call.get("args", {})

            print(f"Executing tool {tool_name} with args: {raw_args}")

            # Extract tool arguments based on the tool type
            if tool_name == "search_papers":
                # Parse and validate search paper args
                tool_args = {
                    "query": raw_args.get("query")
                    or raw_args.get("parameters", {}).get("query"),
                    "limit": raw_args.get("limit", 5),
                    "fields": raw_args.get("fields"),
                }
                return search_papers(**tool_args)

            elif tool_name == "get_single_paper_recommendations":
                # Parse and validate single paper recommendation args
                tool_args = {
                    "paper_id": raw_args.get("paper_id"),
                    "limit": raw_args.get("limit", 5),
                }
                return get_single_paper_recommendations(**tool_args)

            elif tool_name == "get_multi_paper_recommendations":
                # Parse and validate multi paper recommendation args
                tool_args = {
                    "paper_ids": raw_args.get("paper_ids", []),
                    "limit": raw_args.get("limit", 5),
                }
                return get_multi_paper_recommendations(**tool_args)

            else:
                raise ValueError(f"Unknown tool: {tool_name}")

        except Exception as e:
            print(f"Tool execution error: {str(e)}")
            traceback.print_exc()
            return {"status": "error", "error": str(e), "papers": []}

    def handle_message(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming messages and route to appropriate tool"""
        try:
            print("\nHandling message...")
            message = state.get("message", "")
            print(f"Received message: {message}")

            try:
                print("Getting LLM response...")
                messages = shared_state.get_chat_history(limit=3)

                print("Sending prompt to LLM:")
                print(f"Message: {message}")
                print(f"Chat history: {messages}")

                # Get LLM response with tool calling
                response = self.chain.invoke(
                    {"input": message, "chat_history": messages}
                )

                print(f"Raw LLM Response: {response}")

                # Extract tool call data
                tool_call = None
                try:
                    if hasattr(response, "tool_calls") and response.tool_calls:
                        tool_call = {
                            "name": "search_papers",  # Default to search if unclear
                            "args": {},
                        }
                        # Extract from tool calls
                        tool_data = response.tool_calls[0]
                        if isinstance(tool_data, dict):
                            if "name" in tool_data:
                                tool_call["name"] = tool_data["name"]
                            if "function" in tool_data:
                                func_data = tool_data["function"]
                                tool_call["name"] = func_data.get(
                                    "name", tool_call["name"]
                                )
                                # Handle arguments
                                if "arguments" in func_data:
                                    args = func_data["arguments"]
                                    if isinstance(args, str):
                                        try:
                                            args = json.loads(args)
                                        except:
                                            args = {"query": message}
                                    tool_call["args"] = args
                            if "args" in tool_data:
                                tool_call["args"].update(tool_data["args"])

                    elif hasattr(response, "content") and response.content:
                        # Try parsing from content
                        content = response.content
                        try:
                            data = json.loads(content)
                            tool_call = {
                                "name": data.get("name", "search_papers"),
                                "args": data.get("parameters", {}),
                            }
                        except json.JSONDecodeError:
                            tool_call = {
                                "name": "search_papers",
                                "args": {"query": message},
                            }
                    else:
                        tool_call = {
                            "name": "search_papers",
                            "args": {"query": message},
                        }

                    # Ensure we have minimum required arguments
                    if (
                        "query" not in tool_call["args"]
                        and tool_call["name"] == "search_papers"
                    ):
                        tool_call["args"]["query"] = message

                except Exception as parse_error:
                    print(f"Error parsing tool call: {str(parse_error)}")
                    tool_call = {"name": "search_papers", "args": {"query": message}}

                print(f"Final tool call: {tool_call}")
                shared_state.set(config.StateKeys.CURRENT_TOOL, tool_call["name"])

                # Execute tool and handle any tool exceptions
                try:
                    result = self.execute_tool(tool_call)
                    print(f"Tool execution result: {result}")

                    # Process results
                    if result and isinstance(result, dict):
                        if result.get("status") == "error":
                            state["error"] = result.get(
                                "message", "Tool execution failed"
                            )
                        else:
                            papers = result.get("papers", [])
                            if papers:
                                state["response"] = self.format_papers_response(papers)
                                shared_state.add_papers(papers)
                            else:
                                state["response"] = result.get(
                                    "message", "No papers found matching your criteria."
                                )
                    else:
                        raise ValueError("Invalid tool execution result")

                except Exception as tool_error:
                    print(f"Tool execution error: {str(tool_error)}")
                    traceback.print_exc()
                    state["error"] = f"Tool execution error: {str(tool_error)}"

            except Exception as e:
                print(f"Error in message handling: {str(e)}")
                traceback.print_exc()
                state["error"] = f"Error processing message: {str(e)}"

            return state

        except Exception as e:
            print(f"Error in handle_message: {str(e)}")
            traceback.print_exc()
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
