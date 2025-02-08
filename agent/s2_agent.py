from typing import List, Dict, Any, Type, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor

from config.config import config
from state.shared_state import shared_state
from tools.s2.search import s2_tools
from utils.llm import llm_manager


class S2AgentState(TypedDict):
    """Type definition for S2 agent state"""

    message: str
    response: str | None
    error: str | None


class SemanticScholarAgent:
    def __init__(self):
        self.llm = llm_manager.llm.bind_tools(s2_tools)
        self.tool_executor = ToolExecutor(s2_tools)

        # Create prompt template without examples (they're in the system prompt)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", config.S2_AGENT_PROMPT),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

        # Create the chain
        self.chain = (
            RunnablePassthrough.assign(history=lambda x: []) | self.prompt | self.llm
        )

    def handle_message(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming messages and route to appropriate tool"""
        try:
            # Get current message and context
            message = state.get("message", "")
            context = shared_state.get_current_context()

            # Add context to message if needed
            if context:
                message += f"\n\nContext:\n{context}"

            # Get initial response with potential tool calls
            response = self.chain.invoke({"input": message})

            # Initialize messages list with the response
            messages = [HumanMessage(content=message), response]

            # Handle any tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    try:
                        # Execute the tool
                        tool_output = self.tool_executor.invoke(
                            tool_call.name, tool_call.args
                        )
                        # Add tool result to messages
                        messages.append(
                            ToolMessage(
                                content=str(tool_output), tool_call_id=tool_call.id
                            )
                        )
                    except Exception as tool_error:
                        error_msg = f"Tool execution error: {str(tool_error)}"
                        messages.append(
                            ToolMessage(
                                content=str({"error": error_msg}),
                                tool_call_id=tool_call.id,
                            )
                        )

                # Get final response after tool execution
                final_response = self.llm.invoke(messages)
                response_content = final_response.content
            else:
                response_content = response.content

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
        workflow = StateGraph(S2AgentState)

        # Add nodes
        workflow.add_node("process_message", self.handle_message)

        # Add edges
        workflow.add_edge("process_message", END)

        # Set entry point
        workflow.set_entry_point("process_message")

        return workflow.compile()


# Create a global instance
s2_agent = SemanticScholarAgent()
