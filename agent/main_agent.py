import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph

from clients.ollama_client import OllamaClient
from state.agent_state import AgentState, AgentStatus
from tools.ollama_tool import OllamaTool
from tools.paper_analyzer_tool import PaperAnalyzerTool
from tools.semantic_scholar_tool import SemanticScholarTool


class MainAgent:
    """Main supervisor agent that orchestrates the research workflow"""

    SYSTEM_PROMPT = """You are Talk2Papers, an intelligent research assistant specialized in academic paper search and analysis. 
    You have access to the Semantic Scholar database through the semantic_scholar_tool.

    Your primary capabilities include:
    1. semantic_scholar_search: Search for academic papers
       - Use for: Finding papers based on topics, authors, or keywords
       - Input: Search queries with optional filters for year, citations
       - Example: "Find recent papers about large language models"

    Follow these steps for EVERY user request:

    1. UNDERSTAND THE REQUEST
       - Carefully analyze what the user is asking for
       - Determine if they need:
           * Paper search (new papers to review)
           * Specific paper information
           * Refinement of previous search
       - Extract key search parameters (topics, authors, years, etc.)

    2. DETERMINE ACTION NEEDED
       - For new paper searches: Use semantic_scholar_search
       - For paper information: Access existing search context
       - For search refinement: Modify previous search parameters

    3. EXECUTE AND RESPOND
       - Use semantic_scholar_search with precise parameters
       - Format results clearly with proper structure
       - Maintain conversation context for follow-ups

    Current conversation context: {context}
    Current state: {state}
    """

    def __init__(
        self,
        model_name: str = "llama3.2:1b-instruct-q3_K_M",
        tools: Optional[List[BaseTool]] = None,
    ):
        """Initialize the agent with tools and state"""
        try:
            print("[DEBUG] Initializing MainAgent")
            self.state = AgentState()

            # Initialize Ollama client
            self._ollama_client = OllamaClient(model_name=model_name)

            # Initialize tools
            if tools is not None:
                print("[DEBUG] Using provided tools")
                self.tools = tools
                # Find tools by type
                self.semantic_scholar_tool = next(
                    (t for t in tools if isinstance(t, SemanticScholarTool)), None
                )
                self.paper_analyzer_tool = next(
                    (t for t in tools if isinstance(t, PaperAnalyzerTool)), None
                )
                self.ollama_tool = next(
                    (t for t in tools if isinstance(t, OllamaTool)), None
                )
            else:
                print("[DEBUG] Creating new tools")
                self.semantic_scholar_tool = SemanticScholarTool(state=self.state)
                self.ollama_tool = OllamaTool(model_name=model_name, state=self.state)
                self.tools = [
                    self.semantic_scholar_tool,
                    self.ollama_tool,
                ]

            print("[DEBUG] MainAgent initialized successfully")

        except Exception as e:
            print(f"[DEBUG] Error initializing MainAgent: {str(e)}")
            raise

    def _create_supervisor_node(self):
        """Create the supervisor node that routes to tools."""

        async def supervisor(state: MessagesState) -> Dict:
            print("[DEBUG] MainAgent: Processing new message in supervisor")

            if not state.get("messages"):
                return {"next": "__end__"}

            # Get the latest message
            last_message = state["messages"][-1]
            message_content = self._extract_message_content(last_message)
            print(f"[DEBUG] Processing message: {message_content}")

            # Determine intent using Ollama
            intent = await self._determine_intent(message_content)
            print(f"[DEBUG] Determined intent: {intent}")

            # Route based on intent
            if intent == "search":
                return {"messages": state["messages"], "next": "semantic_scholar_tool"}
            elif intent == "analyze":
                return {"messages": state["messages"], "next": "paper_analyzer_tool"}
            else:
                return {"messages": state["messages"], "next": "ollama_tool"}

        return supervisor

    def _create_tool_node(self, tool: Any):
        """Create a node for a specific tool."""

        async def tool_node(state: MessagesState) -> Dict:
            try:
                print(f"[DEBUG] Executing tool: {tool.name}")

                # Extract message content
                last_message = state["messages"][-1]
                message_content = self._extract_message_content(last_message)

                # Update tool state
                tool.set_state(self.state)

                # Execute tool
                result = await tool._arun(message_content)

                print(
                    f"[DEBUG] Tool execution successful, result length: {len(result)}"
                )

                # Update state
                if isinstance(result, dict):
                    self.state.update_state(**result)

                return {
                    "messages": [
                        *state["messages"],
                        {"role": "assistant", "content": result},
                    ],
                    "next": "__end__",
                }

            except Exception as e:
                error_msg = f"Error executing {tool.name}: {str(e)}"
                print(f"[DEBUG] {error_msg}")
                return {
                    "messages": [
                        *state["messages"],
                        {"role": "assistant", "content": error_msg},
                    ],
                    "next": "__end__",
                }

        return tool_node

    def _create_graph(self) -> StateGraph:
        """Create the workflow graph."""
        print("[DEBUG] Creating workflow graph")
        workflow = StateGraph(MessagesState)

        # Add nodes
        workflow.add_node("supervisor", self._create_supervisor_node())
        workflow.add_node(
            "semantic_scholar_tool", self._create_tool_node(self.semantic_scholar_tool)
        )
        workflow.add_node(
            "paper_analyzer_tool", self._create_tool_node(self.paper_analyzer_tool)
        )
        workflow.add_node("ollama_tool", self._create_tool_node(self.ollama_tool))

        # Add edges
        workflow.add_edge(START, "supervisor")
        workflow.add_edge("supervisor", "semantic_scholar_tool")
        workflow.add_edge("supervisor", "paper_analyzer_tool")
        workflow.add_edge("supervisor", "ollama_tool")
        workflow.add_edge("semantic_scholar_tool", "__end__")
        workflow.add_edge("paper_analyzer_tool", "__end__")
        workflow.add_edge("ollama_tool", "__end__")

        print("[DEBUG] Workflow graph created")
        return workflow.compile()

    async def _determine_intent(
        self, message: str, context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Determine the intent of a message using LLM with enhanced classification"""
        try:
            print(f"[DEBUG] MainAgent: Analyzing intent for message: {message}")

            # Build context string
            context_str = ""
            if context:
                if context.get("conversation_history"):
                    context_str += (
                        f"\nConversation history:\n{context['conversation_history']}"
                    )
                if context.get("current_search_results"):
                    context_str += "\nHas current search results: Yes"
                if context.get("focused_paper"):
                    context_str += (
                        f"\nCurrently focused on paper: {context['focused_paper']}"
                    )

            intent_prompt = f"""Analyze this message in context and classify the intent. Return a JSON object.

    Message: "{message}"
    {context_str}

    Rules for classification:
    1. "conversation" if:
       - Asking for explanations (e.g., "explain", "tell me about", "what is")
       - General questions about concepts
       - Asking about previous results or conversation
       - Asking for clarification
       
    2. "search" if:
       - Explicitly requesting papers (e.g., "find papers", "search for", "look for")
       - Searching for specific authors or topics
       - Looking for publications in a time range
       - Needs new search results

    Return JSON in this format:
    {{
        "intent": "search" | "conversation",
        "explanation": "<brief reason>",
        "requires_context": true | false,
        "search_params": {{ }} (only for search intent)
    }}"""

            # Generate response
            response = await self._ollama_client.generate(
                prompt=intent_prompt,
                system_prompt="Return only valid JSON with exact format shown.",
                temperature=0.1,
            )

            # Clean and parse response
            try:
                clean_response = self._clean_json_response(response)
                parsed_response = json.loads(clean_response)
                print(f"[DEBUG] Successfully parsed intent response: {parsed_response}")

                # Add search parameters for search intents
                if parsed_response["intent"] == "search":
                    parsed_response["search_params"] = self._extract_search_params(
                        message
                    )

                return parsed_response

            except json.JSONDecodeError as e:
                print(
                    f"[DEBUG] JSON parse error: {str(e)}\nResponse was: {clean_response}"
                )
                return {
                    "intent": "conversation",
                    "explanation": "Failed to parse intent, defaulting to conversation",
                    "requires_context": False,
                    "search_params": {},
                }

        except Exception as e:
            print(f"[DEBUG] Error determining intent: {str(e)}")
            return {
                "intent": "conversation",
                "explanation": f"Error in intent analysis: {str(e)}",
                "requires_context": False,
                "search_params": {},
            }

    def _clean_json_response(self, response: str) -> str:
        """Clean the response to extract only valid JSON"""
        try:
            # Find the first '{' and last '}'
            start = response.find("{")
            end = response.rfind("}")

            if start != -1 and end != -1:
                return response[start : end + 1]
            return response
        except Exception as e:
            print(f"[DEBUG] Error cleaning JSON response: {str(e)}")
            return response

    def _extract_search_params(self, message: str) -> Dict[str, Any]:
        """Extract search parameters with improved pattern matching"""
        params = {
            "query": message,
            "year_start": None,
            "year_end": datetime.now().year,
            "min_citations": None,
            "author": None,
        }

        # Extract years with improved patterns
        year_patterns = [
            (r"(?:since|after|from)\s+(\d{4})", "year_start"),
            (r"(?:before|until|to)\s+(\d{4})", "year_end"),
            (r"in\s+(\d{4})", "year_exact"),  # For exact year matches
            (r"past\s+(\d+)\s+years?", "year_relative"),  # For relative year ranges
            (r"between\s+(\d{4})\s+and\s+(\d{4})", "year_range"),  # For explicit ranges
        ]

        for pattern, param_type in year_patterns:
            matches = re.finditer(pattern, message, re.IGNORECASE)
            for match in matches:
                if param_type == "year_start":
                    params["year_start"] = int(match.group(1))
                elif param_type == "year_end":
                    params["year_end"] = int(match.group(1))
                elif param_type == "year_exact":
                    exact_year = int(match.group(1))
                    params["year_start"] = exact_year
                    params["year_end"] = exact_year
                elif param_type == "year_relative":
                    years_back = int(match.group(1))
                    params["year_start"] = datetime.now().year - years_back
                elif param_type == "year_range":
                    params["year_start"] = int(match.group(1))
                    params["year_end"] = int(match.group(2))

        # Extract citation requirements
        citation_patterns = [
            r"(?:at least|minimum|min|>)\s*(\d+)\s+citations",
            r"cited\s+(?:at least|minimum|min|>)\s*(\d+)\s+times",
            r"(\d+)\+\s+citations",
        ]

        for pattern in citation_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                params["min_citations"] = int(match.group(1))
                break

        # Extract author if present (improved author detection)
        author_patterns = [
            r"by\s+([A-Z][A-Za-z\s\.-]+)(?=\s|$|\.|,)",
            r"author\s+([A-Z][A-Za-z\s\.-]+)(?=\s|$|\.|,)",
            r"from\s+([A-Z][A-Za-z\s\.-]+)(?=\s|$|\.|,)",
        ]

        for pattern in author_patterns:
            match = re.search(pattern, message)
            if match:
                author = match.group(1).strip()
                # Don't capture common words that might follow "by"
                if author.lower() not in ["the", "a", "an", "this", "that"]:
                    params["author"] = author
                    # Modify query to use author search
                    params["query"] = f'author:"{author}"'
                    break

        return params

    def process_intent(self, intent_data: Dict[str, Any]) -> str:
        """Process the determined intent and route to appropriate handler"""
        intent = intent_data["intent"]

        if intent == "search":
            return "semantic_scholar_tool"
        elif intent == "analysis":
            return "paper_analyzer_tool"
        else:  # conversation
            return "ollama_tool"

    def _extract_message_content(self, message: Any) -> str:
        """Safely extract content from different message types."""
        if isinstance(message, (HumanMessage, SystemMessage, AIMessage)):
            return message.content
        elif isinstance(message, dict):
            return message.get("content", "")
        else:
            return str(message)

    async def process_request(self, message: str) -> AgentState:
        """Process a user request"""
        try:
            print(f"[DEBUG] MainAgent: Processing request: {message}")

            # Add user message to state
            self.state.add_message("user", message)

            # Determine intent with enhanced analysis
            intent_analysis = await self._determine_intent(message)
            print(f"[DEBUG] Intent analysis result: {intent_analysis}")

            # Route based on intent
            if intent_analysis["intent"].startswith("search"):
                print("[DEBUG] MainAgent: Invoking semantic_scholar_tool for search")
                search_result = await self.semantic_scholar_tool._arun(
                    query=intent_analysis["search_params"]["query"],
                    year_start=intent_analysis["search_params"].get("year_start"),
                    year_end=intent_analysis["search_params"].get("year_end"),
                    min_citations=intent_analysis["search_params"].get("min_citations"),
                )

                # Format and add results to state
                if isinstance(search_result, dict):
                    formatted_result = self._format_search_results(search_result)
                    self.state.add_message("system", formatted_result)
                    print(
                        f"[DEBUG] Search results added to state: {len(search_result.get('papers', []))} papers found"
                    )
                else:
                    self.state.add_message("system", str(search_result))
                    print("[DEBUG] Search results added to state (non-dict response)")
            else:
                print("[DEBUG] MainAgent: Processing as conversation")
                result = await self.ollama_tool._arun(message)
                self.state.add_message("system", result)

            self.state.status = AgentStatus.SUCCESS
            return self.state

        except Exception as e:
            print(f"[DEBUG] Error in MainAgent process_request: {str(e)}")
            self.state.status = AgentStatus.ERROR
            self.state.error_message = str(e)
            self.state.add_message(
                "system", f"I apologize, but I encountered an error: {str(e)}"
            )
            return self.state

    def _format_search_results(self, results: Dict) -> str:
        """Format search results for display"""
        if results["status"] == "error":
            return f"Error performing search: {results['error']}"

        if not results.get("papers"):
            return "No papers found matching your criteria."

        formatted_parts = [
            f"Found {results.get('total_results', 0)} papers. Here are the most relevant:"
        ]

        for i, paper in enumerate(results.get("papers", []), 1):
            paper_info = [
                f"\n{i}. {paper.get('title', 'Untitled')}",
                f"Authors: {', '.join(paper.get('authors', []))}",
                f"Year: {paper.get('year', 'N/A')} | Citations: {paper.get('citations', 0)}",
            ]

            if paper.get("abstract"):
                paper_info.append(
                    f"Abstract: {paper['abstract'][:300]}..."
                    if len(paper["abstract"]) > 300
                    else f"Abstract: {paper['abstract']}"
                )

            if paper.get("url"):
                paper_info.append(f"URL: {paper['url']}")

            formatted_parts.extend(paper_info)

        return "\n".join(formatted_parts)

    async def check_health(self) -> Dict[str, bool]:
        """Check health of all components"""
        try:
            return {
                "semantic_scholar": await self.semantic_scholar_tool.check_health(),
                "paper_analyzer": await self.paper_analyzer_tool.check_health(),
                "ollama": await self.ollama_tool.check_health(),
                "main_agent": True,
            }
        except Exception as e:
            return {
                "semantic_scholar": False,
                "paper_analyzer": False,
                "ollama": False,
                "main_agent": False,
                "error": str(e),
            }
