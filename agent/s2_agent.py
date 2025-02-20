import json
from typing import Any, Dict, List, Type, TypedDict
import re
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor
import traceback

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

            # Store the tools
            self.search_tool = s2_tools[0]
            self.single_rec_tool = s2_tools[1]
            self.multi_rec_tool = s2_tools[2]

            # Configure LLM with lower temperature for more consistent responses
            self.llm = llm_manager.llm.bind(
                temperature=0.1,
                stop=["}\n", "}\n\n"],  # Ensure complete JSON
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )

            # Bind tools to LLM
            self.llm = self.llm.bind_tools(s2_tools)

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

    def format_papers_response(self, papers: List[Dict[str, Any]]) -> str:
        """Format papers list into readable response"""
        if not papers:
            return "No papers found matching your query."

        response_parts = ["Here are the relevant papers I found:\n"]

        for i, paper in enumerate(papers, 1):
            # Safely get values with defaults
            title = paper.get("title", "Untitled")
            authors = paper.get("authors", [])
            year = paper.get("year", "N/A")
            cite_count = paper.get("citationCount", 0)
            venue = paper.get("venue", "")
            abstract = paper.get("abstract", "")
            pdf_link = paper.get("openAccessPdf", {})
            if pdf_link:  # Check if not None before getting url
                pdf_url = pdf_link.get("url", "")
            else:
                pdf_url = ""

            # Format authors
            author_names = [a.get("name", "") for a in authors if a]
            author_str = ", ".join(author_names[:3])
            if len(author_names) > 3:
                author_str += " et al."

            # Build paper entry
            entry = [f"\n{i}. {title}"]
            entry.append(f"   Authors: {author_str}")
            entry.append(f"   Year: {year}")

            if cite_count:
                entry.append(f"   Citations: {cite_count}")

            if venue:
                entry.append(f"   Venue: {venue}")

            if abstract:
                # Truncate abstract if too long
                short_abstract = (
                    abstract[:200] + "..." if len(abstract) > 200 else abstract
                )
                entry.append(f"   Abstract: {short_abstract}")

            if pdf_url:
                entry.append(f"   PDF: {pdf_url}")

            response_parts.append("\n".join(entry))

        return "\n".join(response_parts)

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
                    tool_call = response.tool_calls[0]
                    tool_name = tool_call.name
                    tool_args = tool_call.args
                elif hasattr(response, "content") and response.content:
                    try:
                        # Parse tool call from content if in JSON format
                        tool_data = json.loads(response.content)
                        tool_call = {
                            "name": tool_data.get("name", "search_papers"),
                            "args": tool_data.get("args", {"query": message}),
                        }
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                    except json.JSONDecodeError:
                        # Default to search if JSON parsing fails
                        tool_name = "search_papers"
                        tool_args = {"query": message}

                if not tool_name:
                    raise ValueError("No valid tool call found in LLM response")

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

    def execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the selected tool"""
        try:
            tool_name = tool_call["name"]
            params = tool_call["parameters"]

            if tool_name == "search_papers":
                return self.search_tool.invoke(
                    params["query"],
                    limit=params.get("limit", 5),
                    fields=params.get("fields"),
                )
            elif tool_name == "get_single_paper_recommendations":
                return self.single_rec_tool.invoke(
                    params["paper_id"], limit=params.get("limit", 5)
                )
            elif tool_name == "get_multi_paper_recommendations":
                return self.multi_rec_tool.invoke(
                    params["paper_ids"], limit=params.get("limit", 5)
                )
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

        except Exception as e:
            print(f"Tool execution error: {str(e)}")
            return {"status": "error", "error": str(e), "papers": []}

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
