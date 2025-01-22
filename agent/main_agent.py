import asyncio
from typing import Any, Dict, List

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
        """Initialize the agent with tools."""
        print("[DEBUG] Initializing MainAgent")
        self.tools = tools
        self.llm = OllamaClient(model_name="llama3.2:1b-instruct-q3_K_M")
        self.graph = self._create_graph()
        print(f"[DEBUG] MainAgent initialized with {len(tools)} tools")

    def _create_supervisor_node(self):
        """Create the supervisor node that routes to tools."""

        def supervisor(state: MessagesState) -> Dict:
            print("[DEBUG] MainAgent: Processing new message in supervisor")

            if not state.get("messages"):
                return {"next": "__end__"}

            message = state["messages"][-1]["content"].lower()

            # Check for conversation patterns first
            if (
                any(
                    word in message for word in ["hi", "hello", "hey", "bye", "goodbye"]
                )
                or "what can you do" in message
            ):
                print("[DEBUG] Routing to conversation handler")
                return {"messages": state["messages"], "next": "conversation"}

            # Check for search intent
            search_indicators = [
                "find",
                "search",
                "look for",
                "papers about",
                "papers on",
                "research on",
            ]
            if any(indicator in message for indicator in search_indicators):
                print("[DEBUG] Routing to semantic_scholar_tool")
                return {"messages": state["messages"], "next": "semantic_scholar_tool"}

            # Default to conversation
            print("[DEBUG] Default routing to conversation")
            return {"messages": state["messages"], "next": "conversation"}

        return supervisor

    def _create_tool_node(self, tool: Any):
        """Create a node for a specific tool."""

        def tool_node(state: MessagesState) -> Dict:
            try:
                print(f"[DEBUG] Executing tool: {tool.name}")

                # Extract message content safely
                last_message = state["messages"][-1]
                if isinstance(last_message, dict):
                    message_content = last_message.get("content", "")
                else:
                    message_content = (
                        last_message.content
                        if hasattr(last_message, "content")
                        else str(last_message)
                    )

                # Create event loop for async operation
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(tool.arun(message_content))
                loop.close()

                print(
                    f"[DEBUG] Tool execution successful, result length: {len(result)}"
                )

                return {
                    "messages": state["messages"]
                    + [{"role": "assistant", "content": result}],
                    "next": "__end__",
                }
            except Exception as e:
                error_msg = f"Error executing {tool.name}: {str(e)}"
                print(f"[DEBUG] {error_msg}")
                return {
                    "messages": state["messages"]
                    + [{"role": "assistant", "content": error_msg}],
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
            "semantic_scholar_tool", self._create_tool_node(self.tools[0])
        )
        workflow.add_node("conversation", self._handle_conversation)

        # Add edges
        workflow.add_edge(START, "supervisor")
        workflow.add_edge("supervisor", "semantic_scholar_tool")
        workflow.add_edge("supervisor", "conversation")
        workflow.add_edge("semantic_scholar_tool", "__end__")
        workflow.add_edge("conversation", "__end__")

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

    def _handle_conversation(self, state: MessagesState) -> Dict:
        """Handle general conversation without tool invocation"""
        print("[DEBUG] In conversation handler")
        message = state["messages"][-1]["content"].lower()

        if any(word in message for word in ["hi", "hello", "hey"]):
            response = self.CHAT_RESPONSES["greeting"]
        elif "what can you do" in message:
            response = self.CHAT_RESPONSES["capabilities"]
        elif any(word in message for word in ["bye", "goodbye"]):
            response = self.CHAT_RESPONSES["farewell"]
        else:
            response = "I'm your research assistant. Would you like to search for papers about a specific topic?"

        return {
            "messages": [
                *state["messages"],
                {"role": "assistant", "content": response},
            ],
            "next": "__end__",
        }

    async def process_request(self, state: AgentState) -> AgentState:
        """Process a user request using the workflow graph."""
        try:
            print("[DEBUG] MainAgent processing request")

            # Convert messages to proper format
            messages = []
            for msg in state.memory.messages:
                if isinstance(msg, dict):
                    messages.append(msg)
                else:
                    messages.append(
                        {
                            "role": msg.role if hasattr(msg, "role") else "user",
                            "content": msg.content
                            if hasattr(msg, "content")
                            else str(msg),
                        }
                    )

            messages_state = {"messages": messages}
            print(f"[DEBUG] Processing with messages state: {messages_state}")

            result = await self.graph.ainvoke(messages_state)

            if isinstance(result, dict) and "messages" in result:
                for msg in result["messages"]:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        state.add_message("system", msg["content"])
                    elif hasattr(msg, "content"):
                        state.add_message("system", msg.content)

            state.status = AgentStatus.SUCCESS
            return state

        except Exception as e:
            print(f"[DEBUG] Error in process_request: {str(e)}")
            state.status = AgentStatus.ERROR
            state.error_message = str(e)
            state.add_message(
                "system", f"I apologize, but I encountered an error: {str(e)}"
            )
            return state
