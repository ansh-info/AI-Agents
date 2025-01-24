from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph

from state.agent_state import AgentState, AgentStatus
from tools.ollama_tool import OllamaTool
from tools.paper_analyzer_tool import PaperAnalyzerTool
from tools.semantic_scholar_tool import SemanticScholarTool


class MainAgent:
    """Main orchestrator agent that manages the research workflow."""

    SYSTEM_PROMPT = """You are Talk2Papers, an academic research assistant that helps users find, analyze, and understand academic papers.
    
    You have access to these specialized tools:
    1. semantic_scholar_tool: Search and retrieve academic papers
       - Use for: Finding papers, getting citations, retrieving metadata
       - Input: Search queries with optional filters
       
    2. paper_analyzer_tool: Analyze paper content
       - Use for: Summarizing papers, analyzing methods, extracting findings
       - Input: Paper ID/index and analysis request
       
    3. ollama_tool: General text generation and understanding
       - Use for: Answering questions, generating explanations
       - Input: User queries with optional context
    
    Your workflow:
    1. UNDERSTAND THE REQUEST
    - Carefully analyze what the user is asking for
    - Determine which tools are needed
    - Plan the sequence of tool usage

    2. USE TOOLS APPROPRIATELY  
    - semantic_scholar_tool for finding papers
    - paper_analyzer_tool for paper questions
    - ollama_tool for general queries
    
    3. MAINTAIN CONTEXT
    - Track papers being discussed
    - Remember previous searches
    - Build on previous interactions

    Current conversation context: {context}
    Current state: {state}
    """

    def __init__(self, model_name: str = "llama3.2:1b-instruct-q3_K_M"):
        """Initialize the agent with tools and state"""
        print("[DEBUG] Initializing MainAgent")

        # Initialize state
        self.state = AgentState()

        # Initialize tools
        self.semantic_scholar_tool = SemanticScholarTool(state=self.state)
        self.paper_analyzer_tool = PaperAnalyzerTool(
            model_name=model_name, state=self.state
        )
        self.ollama_tool = OllamaTool(model_name=model_name, state=self.state)

        # Create workflow graph
        self.graph = self._create_graph()
        print("[DEBUG] MainAgent initialized with tools and graph")

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
        """Determine the intent of a message using Ollama."""
        try:
            intent_prompt = f"""Analyze this user message and determine the intent:
Message: "{message}"

Possible intents:
1. search - User wants to find papers
2. analyze - User wants to analyze specific papers
3. conversation - General questions or chat

Return only one word (search/analyze/conversation)."""

            intent = await self.ollama_tool._arun(
                prompt=intent_prompt,
                system_prompt="You are an intent classifier. Return only one word.",
                temperature=0.1,
            )

            return intent.strip().lower()

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

    async def process_request(self, state: AgentState) -> AgentState:
        """Process a user request using the workflow graph."""
        try:
            print("[DEBUG] MainAgent processing request")

            # Update local state
            self.state = state

            # Convert messages to proper format
            messages = []
            for msg in state.memory.messages:
                messages.append(
                    {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                )

            # Process through graph
            result = await self.graph.ainvoke({"messages": messages})

            # Update state with results
            if isinstance(result, dict) and "messages" in result:
                for msg in result["messages"]:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        state.add_message("system", msg["content"])

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

    async def check_health(self) -> Dict[str, bool]:
        """Check health of all components."""
        try:
            return {
                "semantic_scholar": await self.semantic_scholar_tool.check_health(),
                "paper_analyzer": await self.paper_analyzer_tool.check_health(),
                "ollama": await self.ollama_tool.check_health(),
                "graph": True,
            }
        except Exception as e:
            return {
                "semantic_scholar": False,
                "paper_analyzer": False,
                "ollama": False,
                "graph": False,
                "error": str(e),
            }
