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
                temperature=0.1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
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
            tool_args = tool_call.get("args", {})

            # Clean up args to match tool expectations
            if "fields" in tool_args:
                # Handle fields if it's a string representation of a list
                if isinstance(tool_args["fields"], str):
                    try:
                        tool_args["fields"] = eval(tool_args["fields"])
                    except:
                        tool_args["fields"] = None

            print(f"Executing tool {tool_name} with args: {tool_args}")

            if tool_name not in self.tools:
                raise ValueError(f"Unknown tool: {tool_name}")

            tool_func = self.tools[tool_name]
            return tool_func(**tool_args)

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

                # Handle tool call formats
                tool_call = None
                if hasattr(response, "tool_calls") and response.tool_calls:
                    # Extract tool call from response
                    tool_dict = response.tool_calls[0]
                    # Handle direct dictionary access
                    tool_name = tool_dict.get("name") or tool_dict.get(
                        "function", {}
                    ).get("name")
                    tool_args = tool_dict.get("args") or tool_dict.get(
                        "function", {}
                    ).get("arguments", {})
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {"query": message}
                elif hasattr(response, "content") and response.content:
                    try:
                        # Parse tool call from content if in JSON format
                        tool_data = json.loads(response.content)
                        tool_name = tool_data.get("name", "search_papers")
                        tool_args = tool_data.get("args", {"query": message})
                    except json.JSONDecodeError:
                        # Default to search if JSON parsing fails
                        tool_name = "search_papers"
                        tool_args = {"query": message}
                else:
                    # Default to search
                    tool_name = "search_papers"
                    tool_args = {"query": message}

                print(f"Final tool call: {tool_name} with args: {tool_args}")
                shared_state.set(config.StateKeys.CURRENT_TOOL, tool_name)

                # Execute tool and handle any tool exceptions
                try:
                    result = self.execute_tool({"name": tool_name, "args": tool_args})
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
