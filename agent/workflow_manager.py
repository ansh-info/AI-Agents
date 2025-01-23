from datetime import datetime
from typing import Dict

from langgraph.graph import START, MessagesState, StateGraph

from agent.main_agent import MainAgent
from state.agent_state import AgentState, AgentStatus
from tools.semantic_scholar_tool import SemanticScholarTool


class ResearchWorkflowManager:
    """
    Manages the research workflow and agent interactions.
    Currently focused on semantic scholar paper search functionality.
    """

    def __init__(self, model_name: str = "llama3.2:1b-instruct-q3_K_M"):
        """Initialize the workflow manager with main agent and tools"""
        # Initialize tools - currently only semantic scholar
        self.tools = [
            SemanticScholarTool(),
        ]

        # Initialize main agent
        self.main_agent = MainAgent(tools=self.tools)

        # Initialize state
        self.current_state = AgentState()

        # Create workflow graph
        self.graph = self._create_workflow_graph()

    def _create_workflow_graph(self) -> StateGraph:
        """Create the workflow graph structure"""
        # Initialize graph with message state
        workflow = StateGraph(MessagesState)

        # Add nodes for main components
        workflow.add_node("supervisor", self.main_agent._create_supervisor_node())

        # Add tool nodes
        for tool in self.tools:
            workflow.add_node(tool.name, self.main_agent._create_tool_node(tool))
            workflow.add_edge("supervisor", tool.name)
            workflow.add_edge(tool.name, "__end__")

        # Add edges
        workflow.add_edge(START, "supervisor")
        workflow.add_edge(
            "supervisor", "__end__"
        )  # Direct path for basic conversations

        return workflow.compile()

        # Create conversation handler node
        def conversation_handler(state: MessagesState) -> Dict:
            """Handle basic conversations"""
            return {
                "messages": state["messages"]
                + [
                    {
                        "role": "assistant",
                        "content": state["messages"][-1].get("assistant_response", ""),
                    }
                ],
                "next": "__end__",
            }

        workflow.add_node("conversation_handler", conversation_handler)

        # Add nodes for main components
        workflow.add_node("supervisor", self.main_agent._create_supervisor_node())

        # Add tool nodes and edges
        for tool in self.tools:
            workflow.add_node(tool.name, self.main_agent._create_tool_node(tool))
            workflow.add_edge("supervisor", tool.name)

        # Add edges
        workflow.add_edge(START, "supervisor")
        workflow.add_edge("supervisor", "conversation_handler")
        workflow.add_edge("conversation_handler", "__end__")
        for tool in self.tools:
            workflow.add_edge(tool.name, "__end__")

        # Set entry point
        workflow.set_entry_point("supervisor")

        return workflow.compile()

    async def _start_node(self, state: MessagesState) -> Dict:
        """Initialize processing of new message"""
        # Update current state
        self.current_state.status = AgentStatus.PROCESSING
        self.current_state.last_update = datetime.now()

        return {"messages": state["messages"], "next": "main_agent"}

    async def _main_agent_node(self, state: MessagesState) -> Dict:
        """Process message through main agent"""
        try:
            # Process through main agent
            result = await self.main_agent.process_request(self.current_state)

            return {"messages": state["messages"] + [result], "next": "update_state"}
        except Exception as e:
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            return {"messages": state["messages"], "next": "update_state"}

    async def _update_state_node(self, state: MessagesState) -> Dict:
        """Update final state after processing"""
        try:
            # Update timestamps and history
            self.current_state.last_update = datetime.now()

            if not hasattr(self.current_state, "state_history"):
                self.current_state.state_history = []

            self.current_state.state_history.append(
                {
                    "timestamp": datetime.now(),
                    "status": self.current_state.status.value,
                    "message_count": len(self.current_state.memory.messages),
                }
            )

            return {"messages": state["messages"], "next": "__end__"}
        except Exception as e:
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            return {"messages": state["messages"], "next": "__end__"}

    async def process_message(self, message: str) -> AgentState:
        """Process a user message through the workflow"""
        try:
            print(f"[DEBUG] Processing message: {message}")

            # Add message to state
            self.current_state.add_message("user", message)

            # Process through graph with simple dict message
            messages_state = {"messages": [{"role": "user", "content": message}]}

            result = await self.graph.ainvoke(messages_state)
            print(f"[DEBUG] Graph result: {result}")

            # Update state with response
            if isinstance(result, dict) and "messages" in result:
                for msg in result["messages"]:
                    # Handle both AIMessage objects and dict messages
                    if hasattr(msg, "content"):  # AIMessage or similar
                        content = msg.content
                        role = "system" if msg.type == "ai" else "user"
                        self.current_state.add_message(role, content)
                        print(
                            f"[DEBUG] Added message from AIMessage: {content[:100]}..."
                        )
                    elif isinstance(msg, dict):
                        if msg.get("role") == "assistant":
                            self.current_state.add_message("system", msg["content"])
                            print(
                                f"[DEBUG] Added message from dict: {msg['content'][:100]}..."
                            )

            self.current_state.status = AgentStatus.SUCCESS
            return self.current_state

        except Exception as e:
            print(f"[DEBUG] Error in workflow: {str(e)}")
            self.current_state.status = AgentStatus.ERROR
            self.current_state.error_message = str(e)
            self.current_state.add_message(
                "system", f"I apologize, but I encountered an error: {str(e)}"
            )
            return self.current_state

    async def check_health(self) -> Dict[str, bool]:
        """Check the health of all components"""
        try:
            # Check main agent
            main_agent_health = True  # Add specific checks if needed

            # Check semantic scholar tool
            s2_tool = self.tools[0]  # Currently only have semantic scholar
            s2_health = await s2_tool.check_health()

            return {
                "main_agent": main_agent_health,
                "semantic_scholar": s2_health,
                "tools": all(tool.check_health() for tool in self.tools),
            }
        except Exception as e:
            return {
                "main_agent": False,
                "semantic_scholar": False,
                "tools": False,
                "error": str(e),
            }
