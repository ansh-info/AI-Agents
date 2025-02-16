import json
from typing import Any, Dict, List, Type, TypedDict
import re
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor

from config.config import config
from state.shared_state import shared_state
from tools.s2 import s2_tools  # Updated import
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

            # Store the search tool
            self.search_tool = s2_tools[0]

            # Configure the LLM with the tool
            self.llm = llm_manager.llm.bind_tools([self.search_tool])

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
        """Generate intelligent default search parameters"""
        # Extract year if present
        year_match = re.search(r"\b(19|20)\d{2}\b", query)
        year = year_match.group() if year_match else None

        # Clean query
        clean_query = re.sub(r"\b(19|20)\d{2}\b", "", query)
        clean_query = re.sub(
            r"\b(published|in|from|year)\b", "", clean_query, flags=re.IGNORECASE
        )
        clean_query = clean_query.strip()

        # Enhance query
        enhanced_query = clean_query
        if year:
            enhanced_query = f"year:{year} {enhanced_query}"

        return {
            "type": "function",
            "name": "search_papers",
            "parameters": {
                "query": enhanced_query,
                "limit": 5,
                "fields": [
                    "paperId",
                    "title",
                    "abstract",
                    "year",
                    "authors",
                    "citationCount",
                    "openAccessPdf",
                    "venue",
                ],
            },
        }

    def parse_tool_call(self, content: str, original_query: str) -> Dict[str, Any]:
        """Parse tool call from response or return default"""
        if not content or content.isspace():
            print("Empty response received, using default parameters")
            return self.get_default_search_params(original_query)

        try:
            # Try to extract just the JSON part if there's additional text
            content = content.strip()
            first_brace = content.find("{")
            last_brace = content.rfind("}")
            if first_brace != -1 and last_brace != -1:
                content = content[first_brace : last_brace + 1]

            # Parse the JSON
            tool_call = json.loads(content)

            # Validate the structure
            if isinstance(tool_call, dict) and tool_call.get("type") == "function":
                name = tool_call.get("name", "")
                params = tool_call.get("parameters", {})

                # Currently only supporting search_papers
                if name != "search_papers":
                    print(f"Tool {name} not yet implemented, using search_papers")
                    return self.get_default_search_params(original_query)

                # Ensure required parameters exist
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
                # Try up to 3 times to get a valid response
                for attempt in range(3):
                    response = self.chain.invoke({"input": message})
                    print(f"LLM Response (attempt {attempt + 1}): {response.content}")

                    if response.content and not response.content.isspace():
                        tool_call = self.parse_tool_call(response.content, message)
                        if tool_call.get("parameters"):
                            break
                else:
                    print("Using enhanced default parameters after retries")
                    tool_call = self.get_default_search_params(message)

                print(f"Final tool call: {tool_call}")
                if "parameters" not in tool_call:
                    raise ValueError("Missing parameters in tool call")

                # Execute search with enhanced parameters
                params = tool_call["parameters"]
                print(f"Executing tool with parameters: {params}")

                search_args = [params["query"]]
                search_kwargs = {
                    "limit": params["limit"],
                    "fields": params.get("fields", None),
                }

                tool_output = self.search_tool.invoke(*search_args, **search_kwargs)
                print(f"Search results: {tool_output}")

                if isinstance(tool_output, dict) and "papers" in tool_output:
                    # Filter results more strictly
                    filtered_papers = []
                    year_match = re.search(r"\b(19|20)\d{2}\b", message)
                    target_year = int(year_match.group()) if year_match else None

                    for paper in tool_output["papers"]:
                        # Skip papers without title or authors
                        if not paper.get("title") or not paper.get("authors"):
                            continue

                        # Apply year filter if specified
                        if target_year and paper.get("year") != target_year:
                            continue

                        filtered_papers.append(paper)

                    response_content = self.format_papers_response(filtered_papers)
                else:
                    response_content = (
                        "Search completed but no matching papers were found."
                    )

                state["response"] = response_content
                state["error"] = None

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
