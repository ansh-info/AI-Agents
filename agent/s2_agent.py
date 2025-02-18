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
            self.search_tool = s2_tools[0]
            self.llm = llm_manager.llm.bind_tools([self.search_tool])

            # Use the prompt from config
            self.prompt = ChatPromptTemplate.from_messages(
                [("system", config.S2_AGENT_PROMPT), ("human", "{input}")]
            )

            self.chain = self.prompt | self.llm
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

            # First check if this is a recommendation request
            paper_ids = self.extract_paper_ids(message)
            if paper_ids:
                if len(paper_ids) == 1:
                    print(f"Processing single paper recommendation for {paper_ids[0]}")
                    tool_call = {
                        "type": "function",
                        "name": "get_single_paper_recommendations",
                        "parameters": {"paper_id": paper_ids[0], "limit": 5},
                    }
                else:
                    print(f"Processing multi-paper recommendation for {paper_ids}")
                    tool_call = {
                        "type": "function",
                        "name": "get_multi_paper_recommendations",
                        "parameters": {
                            "paper_ids": paper_ids[:2],  # Take first two IDs
                            "limit": 5,
                        },
                    }
            else:
                # Regular search query
                print("Processing search query...")
                enhanced_query = self.enhance_query(message)
                tool_call = {
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

            print(f"Final tool call: {tool_call}")

            # Execute the appropriate tool
            if tool_call["name"] == "search_papers":
                result = self.search_tool.invoke(
                    tool_call["parameters"]["query"],
                    limit=tool_call["parameters"]["limit"],
                    fields=tool_call["parameters"].get("fields"),
                )
            elif tool_call["name"] == "get_single_paper_recommendations":
                result = self.get_paper_recommendations(
                    tool_call["parameters"]["paper_id"],
                    limit=tool_call["parameters"]["limit"],
                )
            elif tool_call["name"] == "get_multi_paper_recommendations":
                result = self.get_multi_paper_recommendations(
                    tool_call["parameters"]["paper_ids"],
                    limit=tool_call["parameters"]["limit"],
                )

            print(f"Tool result: {result}")

            # Process the results
            if isinstance(result, dict):
                if result.get("status") == "success":
                    papers = result.get("papers", [])
                    if papers:
                        response_content = self.format_papers_response(papers)
                        state["response"] = response_content
                        shared_state.add_papers(papers)
                    else:
                        state["response"] = "No papers found matching your query."
                else:
                    state["error"] = result.get("error", "Unknown error occurred")
            else:
                state["error"] = "Invalid response format from tool"

            return state

        except Exception as e:
            print(f"Error in handle_message: {str(e)}")
            state["error"] = f"Error in S2 agent: {str(e)}"
            shared_state.set(config.StateKeys.ERROR, state["error"])
            return state

    def extract_paper_ids(self, message: str) -> List[str]:
        """Extract paper IDs from the message"""
        # Look for 40-character hexadecimal IDs
        pattern = r"[a-f0-9]{40}"
        return re.findall(pattern, message.lower())

    def enhance_query(self, query: str) -> str:
        """Enhance search query with domain-specific terms"""
        # Extract year if present
        year_match = re.search(r"\b(19|20)\d{2}\b", query)
        year = year_match.group() if year_match else None

        # Clean query
        clean_query = query.lower()
        clean_query = re.sub(
            r"\b(find|search|get|about|papers|in|from|year|published)\b",
            "",
            clean_query,
        )
        clean_query = re.sub(r"\b(19|20)\d{2}\b", "", clean_query)
        clean_query = clean_query.strip()

        # Add domain-specific terms based on content
        domain_terms = {
            "machine learning": " neural networks deep learning artificial intelligence",
            "neural network": " deep learning machine learning artificial intelligence",
            "computer vision": " image processing object detection convolutional neural networks",
            "nlp": " natural language processing transformers language models",
            "quantum": " quantum computing quantum algorithms quantum supremacy",
            "transformers": " attention mechanism natural language processing bert gpt",
            "deep learning": " neural networks machine learning artificial intelligence",
        }

        for key, terms in domain_terms.items():
            if key in clean_query:
                clean_query += terms

        # Add year if specified
        if year:
            clean_query = f"year:{year} {clean_query}"

        # Add recency terms if no year specified
        if not year and "recent" in query.lower():
            clean_query += " latest developments advances"

        return clean_query.strip()

    def parse_tool_call(self, content: str, original_query: str) -> Dict[str, Any]:
        """Parse tool call from response or return default"""
        if not content or content.isspace():
            print("Empty response received, using enhanced query")
            query = self.enhance_query(original_query)
            return {
                "type": "function",
                "name": "search_papers",
                "parameters": {
                    "query": query,
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

        try:
            # Clean up the response and extract JSON
            content = content.strip()
            first_brace = content.find("{")
            last_brace = content.rfind("}")

            if first_brace != -1 and last_brace != -1:
                json_str = content[first_brace : last_brace + 1]
                tool_call = json.loads(json_str)

                if (
                    isinstance(tool_call, dict)
                    and tool_call.get("type") == "function"
                    and tool_call.get("name") == "search_papers"
                ):

                    # Enhance the query
                    params = tool_call.get("parameters", {})
                    params["query"] = self.enhance_query(
                        params.get("query", original_query)
                    )

                    # Ensure required fields
                    if "limit" not in params:
                        params["limit"] = 5
                    if "fields" not in params:
                        params["fields"] = [
                            "paperId",
                            "title",
                            "abstract",
                            "year",
                            "authors",
                            "citationCount",
                            "openAccessPdf",
                            "venue",
                        ]

                    tool_call["parameters"] = params
                    return tool_call

            return self.get_default_search_params(original_query)

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {str(e)}")
            return self.get_default_search_params(original_query)

    def get_default_search_params(self, query: str) -> Dict[str, Any]:
        """Generate default search parameters with enhanced query"""
        return {
            "type": "function",
            "name": "search_papers",
            "parameters": {
                "query": self.enhance_query(query),
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

    def create_graph(self) -> StateGraph:
        """Create the agent's workflow graph"""
        workflow = StateGraph(S2AgentState)
        workflow.add_node("process_message", self.handle_message)
        workflow.add_edge("process_message", END)
        workflow.set_entry_point("process_message")
        return workflow.compile()


# Create a global instance
s2_agent = SemanticScholarAgent()
