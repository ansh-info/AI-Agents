from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from clients.ollama_client import OllamaClient
from state.agent_state import AgentState, AgentStatus


class MainAgent:
    """Main orchestrator agent that manages the research workflow."""

    SYSTEM_PROMPT = """You are Talk2Papers, an academic research assistant that helps users find, analyze, and understand academic papers.
    You have access to several specialized tools:
    
    1. semantic_scholar_tool: Search and retrieve academic papers
       - Use for: Finding papers, getting citations, retrieving metadata
       - Input: Search queries, paper IDs
       
    2. pdf_tool: Analyze and extract from PDFs
       - Use for: Reading papers, extracting sections
       - Input: Paper URLs, DOIs
       
    3. analysis_tool: In-depth paper analysis  
       - Use for: Summarizing, comparing papers, answering questions
       - Input: Paper content, specific questions

    Your workflow:
    1. UNDERSTAND THE REQUEST
    - Carefully analyze what the user is asking for
    - Determine which tools are needed
    - Plan the sequence of tool usage

    2. USE TOOLS APPROPRIATELY  
    - semantic_scholar_tool for finding papers
    - pdf_tool for reading papers
    - analysis_tool for paper questions
    
    3. MAINTAIN CONTEXT
    - Track papers being discussed
    - Remember previous searches
    - Build on previous interactions

    4. PROVIDE CLEAR RESPONSES
    - Summarize key information
    - Highlight important findings
    - Make connections between papers

    Think step by step about:
    1. What is the user asking for?
    2. Which tools do I need?
    3. In what order should I use them?
    4. How do I combine their outputs?

    Current conversation context: {context}
    Current state: {state}
    """

    def __init__(self, tools: List[Any]):
        """Initialize the agent with tools and model."""
        self.tools = tools
        self.llm = OllamaClient(model_name="llama3.2:1b-instruct-q3_K_M")
        self.graph = self._create_graph()

    def _create_supervisor_node(self):
        """Create the supervisor node that routes to tools."""

        async def supervisor(state: MessagesState) -> Dict:
            # Get conversation history and current state
            messages = state["messages"]
            current_state = state.get("current_state", AgentState())

            # Format context for system prompt
            context = self._format_context(messages)
            state_summary = self._format_state(current_state)

            # Create full prompt
            prompt = self.SYSTEM_PROMPT.format(context=context, state=state_summary)

            # Get LLM decision on next action
            response = await self.llm.generate(
                prompt=messages[-1].content, system_prompt=prompt
            )

            # Parse tool selection
            selected_tool = self._parse_tool_selection(response)

            return {"next": selected_tool}

        return supervisor

    def _create_tool_node(self, tool: Any):
        """Create a node for a specific tool."""

        async def tool_node(state: MessagesState) -> Dict:
            try:
                # Execute tool
                result = await tool.arun(state["messages"][-1].content)

                # Update state
                return {
                    "messages": state["messages"] + [HumanMessage(content=result)],
                    "next": "supervisor",  # Always return to supervisor
                }
            except Exception as e:
                return {
                    "messages": state["messages"]
                    + [HumanMessage(content=f"Error executing tool: {str(e)}")],
                    "next": "supervisor",
                }

        return tool_node

    def _create_graph(self) -> StateGraph:
        """Create the workflow graph."""
        # Create graph
        workflow = StateGraph(MessagesState)

        # Add supervisor node
        workflow.add_node("supervisor", self._create_supervisor_node())

        # Add tool nodes
        for tool in self.tools:
            workflow.add_node(tool.name, self._create_tool_node(tool))

        # Add edges
        workflow.add_edge("supervisor", list(tool.name for tool in self.tools))
        for tool in self.tools:
            workflow.add_edge(tool.name, "supervisor")

        return workflow.compile()

    def _format_context(self, messages: List[Dict]) -> str:
        """Format conversation context for prompt."""
        return "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in messages[-5:]  # Last 5 messages
        )

    def _format_state(self, state: AgentState) -> str:
        """Format current state for prompt."""
        return f"""
        Status: {state.status}
        Search Results: {len(state.search_context.results) if state.search_context else 0} papers
        Focused Paper: {state.memory.focused_paper.title if state.memory.focused_paper else 'None'}
        """

    def _parse_tool_selection(self, llm_response: str) -> str:
        """Parse LLM response to determine tool selection."""
        # Add logic to parse LLM response and determine which tool to use
        # For now, return a simple tool name
        return "semantic_scholar_tool"

    async def process_request(self, state: AgentState) -> AgentState:
        """Process a user request using the workflow graph."""
        try:
            # Convert state to messages format
            messages_state = {"messages": state.memory.messages, "current_state": state}

            # Run through graph
            result = await self.graph.arun(messages_state)

            # Update state with results
            state.memory.messages = result["messages"]
            state.status = AgentStatus.SUCCESS

            return state

        except Exception as e:
            state.status = AgentStatus.ERROR
            state.error_message = str(e)
            return state
