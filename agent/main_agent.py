from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from clients.ollama_client import OllamaClient
from state.agent_state import AgentState, AgentStatus


class MainAgent:
    """Main orchestrator agent that manages the research workflow."""

    SYSTEM_PROMPT = """You are Talk2Papers, an academic research assistant that helps users find, analyze, and understand academic papers.
    You have access to several specialized tools to help with research tasks:
    
    1. search_papers: Search for academic papers on any topic
    2. get_paper_details: Get detailed information about a specific paper
    
    When processing user requests:
    1. UNDERSTAND THE REQUEST
    - Carefully analyze what the user is asking for
    - Determine if they need paper search, details, analysis, etc.
    
    2. USE APPROPRIATE TOOLS
    - For finding papers: Use search_papers
    - For paper details: Use get_paper_details
    - Use tools in logical sequence
    
    3. MAINTAIN CONTEXT
    - Track papers being discussed
    - Remember previous searches
    - Connect related information
    
    4. PROVIDE CLEAR RESPONSES
    - Summarize key information
    - Highlight important findings
    - Make recommendations when appropriate
    
    Always think step by step and explain your reasoning when using tools."""

    def __init__(
        self, tools: List[BaseTool], model_name: str = "llama3.2:1b-instruct-q3_K_M"
    ):
        """Initialize the agent with tools and model."""
        self.tools = tools
        self.llm = OllamaClient(model_name=model_name)
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """Create the agent's workflow graph."""
        workflow = StateGraph(MessagesState)

        # Add nodes
        workflow.add_node("start", self._start_node)
        workflow.add_node("router", self._route_request)
        workflow.add_node("process_search", self._process_search)
        workflow.add_node("process_details", self._process_details)
        workflow.add_node("update_memory", self._update_memory)

        # Add edges
        workflow.add_edge(START, "start")
        workflow.add_edge("start", "router")
        workflow.add_edge("router", "process_search")
        workflow.add_edge("router", "process_details")
        workflow.add_edge("process_search", "update_memory")
        workflow.add_edge("process_details", "update_memory")
        workflow.add_edge("update_memory", END)

        # Set entry point
        workflow.set_entry_point("start")

        return workflow.compile()

    async def process_request(self, state: AgentState) -> AgentState:
        """Process a user request using the workflow graph."""
        try:
            result = await self.graph.arun(
                {"messages": state.memory.messages, "current_state": state}
            )
            return result["current_state"]
        except Exception as e:
            state.status = AgentStatus.ERROR
            state.error_message = str(e)
            return state

    def _start_node(self, state: MessagesState) -> Dict:
        """Initialize processing of a new request."""
        return {"messages": state["messages"], "next": "router"}

    def _route_request(self, state: MessagesState) -> Dict:
        """Route the request to appropriate handler based on user intent."""
        last_message = state["messages"][-1].content.lower()

        # Determine intent through LLM
        system_prompt = "Determine if this request is about: 1) searching for papers, or 2) getting paper details."
        try:
            response = self.llm.generate(
                prompt=last_message, system_prompt=system_prompt, max_tokens=50
            )

            if "search" in response.lower():
                return {"next": "process_search"}
            else:
                return {"next": "process_details"}
        except Exception as e:
            # Default to search if there's an error
            return {"next": "process_search"}

    async def _process_search(self, state: MessagesState) -> Dict:
        """Process paper search requests."""
        try:
            message = state["messages"][-1].content
            print(f"[DEBUG] Processing search request: {message}")

            # Use search tool
            for tool in self.tools:
                if hasattr(tool, "search_papers"):
                    result = await tool.search_papers(query=message)
                    print(
                        f"[DEBUG] Search result: {result[:200]}..."
                    )  # Print first 200 chars
                    return {
                        "messages": state["messages"] + [HumanMessage(content=result)],
                        "next": "update_memory",
                    }

            print("[DEBUG] Search tool not found")
            return {
                "messages": state["messages"]
                + [HumanMessage(content="Search tool not available")],
                "next": "update_memory",
            }
        except Exception as e:
            print(f"[DEBUG] Search error: {str(e)}")
            return {
                "messages": state["messages"]
                + [HumanMessage(content=f"Error in search: {str(e)}")],
                "next": "update_memory",
            }

    async def _route_request(self, state: MessagesState) -> Dict:
        """Route the request to appropriate handler based on user intent."""
        try:
            last_message = state["messages"][-1].content
            print(f"[DEBUG] Routing message: {last_message}")

            # Use LLM to determine intent
            intent_prompt = f"""Analyze this message and determine the user's intent:
Message: {last_message}

Classify the intent as one of:
1. search - User wants to find papers on a topic
2. paper_question - User is asking about a specific paper
3. comparison - User wants to compare papers
4. conversation - General conversation or other intent

Respond with just the intent category."""

            intent_response = await self.llm.generate(
                prompt=intent_prompt,
                system_prompt="You are a research assistant intent classifier. Respond with only the intent category.",
                max_tokens=50,
            )

            print(f"[DEBUG] Detected intent: {intent_response.strip()}")

            # Route based on intent
            intent = intent_response.strip().lower()
            if "search" in intent:
                return {"next": "process_search"}
            elif "paper_question" in intent:
                return {"next": "process_details"}
            elif "comparison" in intent:
                return {"next": "process_comparison"}
            else:
                return {"next": "process_conversation"}

        except Exception as e:
            print(f"[DEBUG] Error in route_request: {str(e)}")
            # Default to conversation on error
            return {"next": "process_conversation"}

    async def _process_details(self, state: MessagesState) -> Dict:
        """Process paper details requests."""
        try:
            message = state["messages"][-1].content

            # Use details tool
            for tool in self.tools:
                if tool.name == "get_paper_details":
                    result = await tool._arun(message)
                    return {
                        "messages": state["messages"] + [HumanMessage(content=result)],
                        "next": "update_memory",
                    }

            return {
                "messages": state["messages"]
                + [HumanMessage(content="Details tool not available")],
                "next": "update_memory",
            }
        except Exception as e:
            return {
                "messages": state["messages"]
                + [HumanMessage(content=f"Error getting details: {str(e)}")],
                "next": "update_memory",
            }

    def _update_memory(self, state: MessagesState) -> Dict:
        """Update conversation memory and state."""
        try:
            # Add any final processing here
            return {"messages": state["messages"], "next": END}
        except Exception as e:
            return {
                "messages": state["messages"]
                + [HumanMessage(content=f"Error updating memory: {str(e)}")],
                "next": END,
            }
