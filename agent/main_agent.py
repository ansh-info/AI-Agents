from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from langgraph.graph import (
    START,  # Updated import
    MessagesState,
    StateGraph,
)

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

    CHAT_RESPONSES = {
        "greeting": "Hello! I'm Talk2Papers, your research assistant. I can help you find and understand academic papers. What would you like to know?",
        "capabilities": """I can help you with:
1. Finding academic papers on any topic
2. Getting detailed information about specific papers
3. Understanding paper content and relationships
What would you like to explore?""",
        "farewell": "Goodbye! Feel free to return whenever you need help with research papers.",
    }

    def __init__(self, tools: List[Any]):
        """Initialize the agent with tools and model."""
        print("[DEBUG] Initializing MainAgent")
        self.tools = tools
        self.llm = OllamaClient(model_name="llama3.2:1b-instruct-q3_K_M")
        self.graph = self._create_graph()
        print(f"[DEBUG] MainAgent initialized with {len(tools)} tools")

    def _create_supervisor_node(self):
        """Create the supervisor node that routes to tools."""

        async def supervisor(state: MessagesState) -> Dict:
            print("[DEBUG] MainAgent: Processing new message in supervisor")

            messages = state["messages"]
            current_state = state.get("current_state", AgentState())

            # Check if this is a conversation message first
            last_message = messages[-1].content
            if any(
                trigger in last_message.lower()
                for trigger in [
                    "hi",
                    "hello",
                    "hey",
                    "what can you do",
                    "bye",
                    "goodbye",
                ]
            ):
                response = self._handle_conversation(last_message)
                return {
                    "messages": messages + [HumanMessage(content=response)],
                    "next": "__end__",
                }

            # Tool selection for research tasks
            context = self._format_context(messages)
            state_summary = self._format_state(current_state)

            prompt = self.SYSTEM_PROMPT.format(context=context, state=state_summary)

            response = await self.llm.generate(
                prompt=messages[-1].content, system_prompt=prompt
            )

            selected_tool = self._parse_tool_selection(response)
            print(f"[DEBUG] Selected tool: {selected_tool}")

            return {"next": selected_tool}

        return supervisor

    def _create_tool_node(self, tool: Any):
        """Create a node for a specific tool."""

        async def tool_node(state: MessagesState) -> Dict:
            try:
                print(f"[DEBUG] Executing tool: {tool.name}")
                result = await tool.arun(state["messages"][-1].content)
                print(
                    f"[DEBUG] Tool execution successful, result length: {len(result)}"
                )

                return {
                    "messages": state["messages"] + [HumanMessage(content=result)],
                    "next": "supervisor",
                }
            except Exception as e:
                error_msg = f"Error executing {tool.name}: {str(e)}"
                print(f"[DEBUG] {error_msg}")
                return {
                    "messages": state["messages"] + [HumanMessage(content=error_msg)],
                    "next": "supervisor",
                }

        return tool_node

    def _create_graph(self) -> StateGraph:
        """Create the workflow graph."""
        print("[DEBUG] Creating workflow graph")
        workflow = StateGraph(MessagesState)

        # Add supervisor node
        workflow.add_node("supervisor", self._create_supervisor_node())

        # Add tool nodes and edges
        for tool in self.tools:
            workflow.add_node(tool.name, self._create_tool_node(tool))
            workflow.add_edge("supervisor", tool.name)
            workflow.add_edge(tool.name, "supervisor")

        # Add START edge correctly using the imported constant
        workflow.add_edge(START, "supervisor")

        print("[DEBUG] Workflow graph created")
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
        Focused Paper: {state.memory.focused_paper.title if state.memory.focused_paper else "None"}
        """

    def _parse_tool_selection(self, llm_response: str) -> str:
        """Parse LLM response to determine tool selection."""
        # For now, defaulting to semantic scholar for search queries
        return "semantic_scholar_tool"

    def _handle_conversation(self, message: str) -> str:
        """Handle general conversation without tool invocation"""
        print(f"[DEBUG] Handling conversation: {message}")
        message_lower = message.lower()

        if any(word in message_lower for word in ["hi", "hello", "hey"]):
            return self.CHAT_RESPONSES["greeting"]
        elif "what can you do" in message_lower:
            return self.CHAT_RESPONSES["capabilities"]
        elif any(word in message_lower for word in ["bye", "goodbye"]):
            return self.CHAT_RESPONSES["farewell"]
        else:
            return "I'm your research assistant. Would you like to search for papers about a specific topic?"

    async def process_request(self, state: AgentState) -> AgentState:
        """Process a user request using the workflow graph."""
        try:
            print("[DEBUG] MainAgent processing request")
            messages_state = {"messages": state.memory.messages, "current_state": state}

            # Change from arun to ainvoke
            result = await self.graph.ainvoke(messages_state)

            state.memory.messages = result["messages"]
            state.status = AgentStatus.SUCCESS

            print("[DEBUG] Request processed successfully")
            return state

        except Exception as e:
            print(f"[DEBUG] Error in process_request: {str(e)}")
            state.status = AgentStatus.ERROR
            state.error_message = str(e)
            return state
