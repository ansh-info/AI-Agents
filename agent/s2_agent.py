from typing import Any, Dict

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor

from config.config import config
from state.shared_state import shared_state
from tools.s2.search import s2_tools
from utils.llm import llm_manager


class SemanticScholarAgent:
    def __init__(self):
        self.llm = llm_manager.llm.bind_tools(s2_tools)
        self.tool_executor = ToolExecutor(s2_tools)

    def handle_message(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming messages and route to appropriate tool"""
        try:
            # Get current message and context
            message = state.get("message", "")
            context = shared_state.get_current_context()

            # Create messages list
            messages = [HumanMessage(content=message)]

            # Get LLM response with potential tool calls
            ai_msg = self.llm.invoke(messages)
            messages.append(ai_msg)

            # Handle any tool calls
            if ai_msg.tool_calls:
                for tool_call in ai_msg.tool_calls:
                    # Execute the tool
                    tool_output = self.tool_executor.invoke(
                        tool_call.name, tool_call.args
                    )
                    # Add tool result to messages
                    messages.append(
                        ToolMessage(content=str(tool_output), tool_call_id=tool_call.id)
                    )

                # Get final response after tool execution
                final_response = self.llm.invoke(messages)
                response_content = final_response.content
            else:
                response_content = ai_msg.content

            # Update state and shared state
            state["response"] = response_content
            shared_state.add_to_chat_history("assistant", response_content)

            return state

        except Exception as e:
            error_msg = f"Error in S2 agent: {str(e)}"
            state["error"] = error_msg
            shared_state.set(config.StateKeys.ERROR, error_msg)
            return state

    def create_graph(self) -> StateGraph:
        """Create the agent's workflow graph"""
        workflow = StateGraph()

        # Add nodes
        workflow.add_node("process_message", self.handle_message)

        # Add edges
        workflow.add_edge("process_message", END)

        # Set entry point
        workflow.set_entry_point("process_message")

        return workflow


# Create a global instance
s2_agent = SemanticScholarAgent()
