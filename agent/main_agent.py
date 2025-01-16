from typing import Any, Dict, List, Optional

from agent_state import AgentState, AgentStatus
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph


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
    
    Always think step by step and explain your reasoning when using tools.
    """

    def __init__(
        self, tools: List[BaseTool], model_name: str = "llama3.2:1b-instruct-q3_K_M"
    ):
        self.tools = tools
        self.llm = ChatOpenAI(model=model_name)
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
        workflow.add_edge("start", "router")
        workflow.add_edge("router", "process_search")
        workflow.add_edge("router", "process_details")
        workflow.add_edge("process_search", "update_memory")
        workflow.add_edge("process_details", "update_memory")

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
        response = self.llm.invoke([system_prompt, last_message])

        if "search" in response.content.lower():
            return {"next": "process_search"}
        else:
            return {"next": "process_details"}

    def _process_search(self, state: MessagesState) -> Dict:
        """Process paper search requests."""
        # Implementation here
        pass

    def _process_details(self, state: MessagesState) -> Dict:
        """Process paper details requests."""
        # Implementation here
        pass

    def _update_memory(self, state: MessagesState) -> Dict:
        """Update conversation memory and state."""
        # Implementation here
        pass
