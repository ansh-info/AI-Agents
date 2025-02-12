from typing import List, Dict, Any, Type, TypedDict
import json
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
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

            # Get the search tool
            self.search_tool = s2_tools[0]

            # Configure the LLM with the tool
            self.llm = llm_manager.llm.bind_tools([self.search_tool])

            # Create tool executor
            self.tool_executor = ToolExecutor(tools=[self.search_tool])

            # Create prompt template
            self.prompt = ChatPromptTemplate.from_messages(
                [("system", config.S2_AGENT_PROMPT), ("human", "{input}")]
            )

            # Create chain
            self.chain = self.prompt | self.llm

            print("S2 Agent initialized successfully")

        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def get_default_search_params(self, query: str) -> Dict[str, Any]:
        """Generate default search parameters"""
        return {
            "type": "function",
            "name": "search_papers",
            "parameters": {"query": f"{query} recent research", "limit": 5},
        }

    def parse_tool_call(self, content: str, original_query: str) -> Dict[str, Any]:
        """Parse tool call from response or return default"""
        if not content or content.isspace():
            print("Empty response received, using default parameters")
            return self.get_default_search_params(original_query)

        try:
            # First try to parse as JSON
            tool_call = json.loads(content)

            if isinstance(tool_call, dict) and tool_call.get("type") == "function":
                # Validate the structure
                params = tool_call.get("parameters", {})
                if not params.get("query"):
                    params["query"] = original_query
                if not params.get("limit"):
                    params["limit"] = 5

                return {
                    "type": "function",
                    "name": "search_papers",
                    "parameters": params,
                }

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing JSON: {e}\nUsing default parameters")

        # If we get here, use default parameters
        return self.get_default_search_params(original_query)

    def format_papers_response(self, papers: List[Dict[str, Any]]) -> str:
        """Format papers list into readable response"""
        if not papers:
            return "No papers found matching your query."

        response = "Here are the relevant papers I found:\n\n"
        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "Untitled")
            authors = paper.get("authors", [])
            year = paper.get("year", "N/A")

            author_names = [a.get("name", "") for a in authors]
            author_str = ", ".join(author_names[:3])
            if len(authors) > 3:
                author_str += " et al."

            response += f"{i}. {title}\n"
            response += f"   Authors: {author_str}\n"
            response += f"   Year: {year}\n\n"

        return response

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

                # Parse tool call or get default parameters
                tool_call = self.parse_tool_call(response.content, message)
                print(f"Parsed tool call: {tool_call}")
                
                if "parameters" not in tool_call:
                    raise ValueError("Missing parameters in tool call")
                
                # Extract parameters
                params = tool_call["parameters"]
                print(f"Executing tool with parameters: {params}")
                
                # Execute search with the proper invocation format
                search_args = [params["query"]]
                search_kwargs = {"limit": params["limit"]}
                tool_output = self.search_tool.invoke(*search_args, **search_kwargs)
                
                print(f"Search results: {tool_output}")

                if isinstance(tool_output, dict) and "papers" in tool_output:
                    response_content = self.format_papers_response(tool_output["papers"])
                else:
                    response_content = "Search completed but no papers were found."

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
            return state    def handle_message(self, state: Dict[str, Any]) -> Dict[str, Any]:
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

                # Parse tool call or get default parameters
                tool_call = self.parse_tool_call(response.content, message)
                print(f"Parsed tool call: {tool_call}")
                
                if "parameters" not in tool_call:
                    raise ValueError("Missing parameters in tool call")
                
                # Extract parameters and run the search directly
                params = tool_call["parameters"]
                print(f"Executing tool with parameters: {params}")
                
                # Execute the search function directly
                tool_output = search_papers(
                    query=params["query"],
                    limit=params["limit"]
                )
                
                print(f"Search results: {tool_output}")

                if isinstance(tool_output, dict) and "papers" in tool_output:
                    response_content = self.format_papers_response(tool_output["papers"])
                else:
                    response_content = "Search completed but no papers were found."

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
