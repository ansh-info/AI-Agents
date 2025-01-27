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

    SYSTEM_PROMPT = """You are Talk2Papers, an academic research assistant designed to help users find, analyze, and understand academic papers. You have access to these specialized tools:

    1. semantic_scholar_search: Search for academic papers
       - Use for: Finding papers based on topics, authors, or keywords
       - Input: Search queries with optional filters for year, citations
       - Example: "Find recent papers about large language models"
       
    2. paper_analyzer: Analyze paper content
       - Use for: Understanding paper content, extracting insights
       - Input: Paper ID and analysis request
       - Example: "Analyze the methodology of paper 1"
       
    3. conversation: Handle general queries and explanations
       - Use for: Questions about results, explanations, clarifications
       - Example: "Explain what this paper is about"

    Approach each request step by step:

    1. UNDERSTAND THE REQUEST
       - Carefully analyze what the user is asking for
       - Identify if they need:
         * Paper search (finding new papers)
         * Paper analysis (understanding specific papers)
         * General help or explanations

    2. DETERMINE TOOLS TO USE
       - Choose the appropriate tool based on the request:
         * Use semantic_scholar_search for finding papers
         * Use paper_analyzer for analyzing specific papers
         * Use conversation for general queries

    3. EXECUTE AND RESPOND
       - Use tools in logical sequence
       - Maintain conversation context
       - Provide clear, structured responses

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
            self.ollama_client = OllamaClient(model_name=model_name)

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
                self.paper_analyzer_tool = PaperAnalyzerTool(
                    model_name=model_name, state=self.state
                )
                self.ollama_tool = OllamaTool(model_name=model_name, state=self.state)
                self.tools = [
                    self.semantic_scholar_tool,
                    self.paper_analyzer_tool,
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

    async def _determine_intent(self, message: str) -> str:
        """Determine the intent of a message using LLM"""
        try:
            intent_prompt = (
                "Classify the following message into one of these categories:\n"
                "- search: looking for papers\n"
                "- analyze: analyzing specific papers\n"
                "- conversation: general chat\n\n"
                f"Message: {message}\n\n"
                "Return ONLY ONE WORD (search/analyze/conversation):"
            )

            response = await self.ollama_client.generate(
                prompt=intent_prompt,
                system_prompt="You are an intent classifier. Return only one word.",
                temperature=0.1,
            )

            # Clean up response
            response = response.strip().lower()
            if "search" in response:
                return "search"
            elif "analyze" in response:
                return "analyze"
            return "conversation"

        except Exception as e:
            print(f"[DEBUG] Error determining intent: {str(e)}")
            return "conversation"

    def _extract_message_content(self, message: Any) -> str:
        """Safely extract content from different message types."""
        if isinstance(message, (HumanMessage, SystemMessage, AIMessage)):
            return message.content
        elif isinstance(message, dict):
            return message.get("content", "")
        else:
            return str(message)

    async def process_request(self, request: str) -> AgentState:
        """Process a request through the workflow"""
        try:
            print(f"[DEBUG] Processing request: {request}")

            # Initialize messages state for graph
            messages_state = {"messages": [HumanMessage(content=request)]}

            # Determine intent first
            intent = await self.main_agent._determine_intent(request)
            print(f"[DEBUG] Determined intent: {intent}")

            if intent == "conversation":
                # For conversation, use OllamaTool directly
                result = await self.ollama_tool._arun(request)
                self.state.add_message("system", result)
            elif intent == "search":
                # For search, use semantic scholar tool
                result = await self.semantic_scholar_tool._arun(request)

                if isinstance(result, dict) and result.get("status") == "success":
                    formatted_result = self._format_search_results(result)
                    self.state.add_message("system", formatted_result)
                else:
                    error_msg = result.get("error", "Unknown search error")
                    self.state.add_message("system", f"Error in search: {error_msg}")
            else:
                # Default to conversation
                result = await self.ollama_tool._arun(request)
                self.state.add_message("system", result)

            # Ensure state is updated
            if self.state.status != AgentStatus.ERROR:
                self.state.status = AgentStatus.SUCCESS

            return self.state

        except Exception as e:
            print(f"[DEBUG] Error processing request: {str(e)}")
            self.state.status = AgentStatus.ERROR
            self.state.error_message = str(e)
            self.state.add_message(
                "system", f"I apologize, but I encountered an error: {str(e)}"
            )
            return self.state

    def _format_search_results(self, results: Dict) -> str:
        """Format search results for display"""
        if results.get("status") == "error":
            return f"Error performing search: {results.get('error')}"

        papers = results.get("papers", [])
        if not papers:
            return "No papers found matching your criteria."

        formatted_parts = [
            f"Found {results.get('total_results', 0)} papers. Here are the most relevant:\n"
        ]

        for i, paper in enumerate(papers, 1):
            paper_info = [
                f"\n{i}. {paper.title}",
                f"Authors: {', '.join(a.name for a in paper.authors)}",
                f"Year: {paper.year or 'N/A'} | Citations: {paper.citations or 0}",
            ]

            if paper.abstract:
                paper_info.append(
                    f"Abstract: {paper.abstract[:300]}..."
                    if len(paper.abstract) > 300
                    else f"Abstract: {paper.abstract}"
                )

            paper_info.append(f"[View Paper]({paper.url})\n")
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
